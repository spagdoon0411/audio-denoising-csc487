from models.unet.unet import OurUNet
from utils.dataset_utils import AudioData
import tensorflow as tf
import sys
import os
from data_paths import data_paths, data_config

USING_DB = True

# Util object for data preprocessing
data = AudioData(clean_audio_path=data_paths["clean"]["train"],
                 clean_audio_test_path=data_paths["clean"]["test"],
                 noisy_audio_path=data_paths["noise"]["train"],
                 noisy_audio_test_path=data_paths["noise"]["test"],
                 sampling_rate=data_config["sample_rate"],
                 hop_length=data_config["hop_length"],
                 noise_level=data_config["noise_level"],
                 clean_vec_repeats=data_config["clean_vec_repeats"])

vec_path = data_paths["vectors"]
print(f"Saving vector Dataset objects in folder {vec_path}")
data.clean_mixed_vectors_train_dataset.save(path=data_paths["vectors"]["train"])
data.clean_mixed_vectors_test_dataset.save(path=data_paths["vectors"]["test"])

def tensor_spectrogram_from_tensor_audio(tens):
    return tf.signal.stft(
        signals=tens,
        frame_length=data_config["frame_length"],
        frame_step=data_config["hop_length"],
        fft_length=data_config["fft_length"],
        window_fn=tf.signal.hann_window,
    )


def power_to_db(S, amin=1e-16, top_db=80.0):
    """Convert a power-spectrogram (magnitude squared) to decibel (dB) units.
    Computes the scaling ``10 * log10(S / max(S))`` in a numerically
    stable way.
    Based on:
    https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
    """
    
    if(not USING_DB):
        return S

    def _tf_log10(x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    # Scale magnitude relative to maximum value in S. Zeros in the output
    # correspond to positions where S == ref.
    ref = tf.reduce_max(S)

    log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
    log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))

    log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

    return log_spec

print("Mapping decibel spectrogram conversion over dataset... ")


# Dataset of training mixed-clean spectrogram pairs 
train_spects: tf.data.Dataset = data.clean_mixed_vectors_train_dataset.map(
    lambda tens1, tens2: (
        tf.expand_dims(power_to_db(tf.abs(tensor_spectrogram_from_tensor_audio(tens1))), axis=-1),
        tf.expand_dims(power_to_db(tf.abs(tensor_spectrogram_from_tensor_audio(tens2))), axis=-1),
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False,
    name="vecstospecttensors"
)

# Dataset of testing mixed-clean spectrogram pairs
test_spects: tf.data.Dataset = data.clean_mixed_vectors_test_dataset.map(
    lambda tens1, tens2: (
        tf.expand_dims(power_to_db(tf.abs(tensor_spectrogram_from_tensor_audio(tens1))), axis=-1),
        tf.expand_dims(power_to_db(tf.abs(tensor_spectrogram_from_tensor_audio(tens2))), axis=-1),
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False,
    name="vecstospecttensors"
)

# Notifies user of the first few sizes in the dataset
print("Some training sizes: ")
for spect1, spect2 in train_spects.take(10):
    print("Mixed size: ", spect1.shape, "Clean size: ", spect2.shape)

print("Some testing sizes: ")
for spect1, spect2 in test_spects.take(10):
    print("Mixed size: ", spect1.shape, "Clean size: ", spect2.shape)

spect_path = data_paths["spectrograms"]
print(f"Saving spectrogram Dataset objects in folder {spect_path}")
train_spects.save(path=data_paths["spectrograms"]["train"])
test_spects.save(path=data_paths["spectrograms"]["test"])

print("Done processing data!")

