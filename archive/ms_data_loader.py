from models.unet.unet import OurUNet
from utils.dataset_utils import AudioData
import tensorflow as tf

path_clean_train = "/Users/spandansuthar/Projects/audio-denoising-csc487/data/MS-SNSD/clean_train"
path_clean_test = "/Users/spandansuthar/Projects/audio-denoising-csc487/data/MS-SNSD/clean_test"
path_noise_train = "/Users/spandansuthar/Projects/audio-denoising-csc487/data/MS-SNSD/noise_train"
path_noise_test = "/Users/spandansuthar/Projects/audio-denoising-csc487/data/MS-SNSD/noise_test"

sample_rate = 16000
hop_length = 256
noise_level = 0.1
clean_vec_repeats = 2

data = AudioData(clean_audio_path=path_clean_train,
                 clean_audio_test_path=path_clean_test,
                 noisy_audio_path=path_noise_train,
                 noisy_audio_test_path=path_noise_test,
                 sampling_rate=sample_rate,
                 hop_length=hop_length,
                 noise_level=noise_level,
                 clean_vec_repeats=clean_vec_repeats)

path_tf_dataset_train = "/Volumes/baseqi1tb/ms_dataset_vectors/train"
path_tf_dataset_test = "/Volumes/baseqi1tb/ms_dataset_vectors/test"

data.clean_mixed_vectors_test_dataset.save(path=path_tf_dataset_test)
data.clean_mixed_vectors_train_dataset.save(path=path_tf_dataset_train)

def tensor_spectrogram_from_tensor_audio(tens):
    return tf.signal.stft(
        signals=tens,
        frame_length=hop_length * 4,
        frame_step=hop_length,
        fft_length=hop_length * 4,
        window_fn=tf.signal.hann_window,
    )

USING_DB = False

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

train_spects: tf.data.Dataset = data.clean_mixed_vectors_train_dataset.map(
    lambda tens1, tens2: (
        tf.expand_dims(power_to_db(tf.abs(tensor_spectrogram_from_tensor_audio(tens1))), axis=-1),
        tf.expand_dims(power_to_db(tf.abs(tensor_spectrogram_from_tensor_audio(tens2))), axis=-1),
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False,
    name="vecstospecttensors"
)
test_spects: tf.data.Dataset = data.clean_mixed_vectors_test_dataset.map(
    lambda tens1, tens2: (
        tf.expand_dims(power_to_db(tf.abs(tensor_spectrogram_from_tensor_audio(tens1))), axis=-1),
        tf.expand_dims(power_to_db(tf.abs(tensor_spectrogram_from_tensor_audio(tens2))), axis=-1),
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
    deterministic=False,
    name="vecstospecttensors"
)

for spect1, spect2 in train_spects.take(10):
    print("Some train sizes: ", spect1.shape)

for spect1, spect2 in test_spects.take(10):
    print("Some test sizes: ", spect2.shape)

path_tf_dataset_spect_train = "/Volumes/baseqi1tb/ms_dataset_spects/train"
path_tf_dataset_spect_test = "/Volumes/baseqi1tb/ms_dataset_spects/test"

train_spects.save(path=path_tf_dataset_spect_train)
test_spects.save(path=path_tf_dataset_spect_test)
