from utils.dataset_utils import AudioData
import tensorflow as tf
from data_paths import data_paths, data_config
import tensorflow as tf

USING_DB = True
NORMALIZE = False

# Util object for data preprocessing
data = AudioData(clean_audio_path=data_paths["clean"]["train"],
                 clean_audio_test_path=data_paths["clean"]["test"],
                 noisy_audio_path=data_paths["noise"]["train"],
                 noisy_audio_test_path=data_paths["noise"]["test"],
                 sampling_rate=data_config["sample_rate"],
                 hop_length=data_config["hop_length"],
                 noise_level=data_config["noise_level"],
                 clean_vec_repeats=data_config["clean_vec_repeats"],
                 frame_length=data_config["frame_length"],
                 fft_length=data_config["fft_length"])

# Use the same SpectUtils instance that's in the AudioData object above; it's contains
# all of the STFT parameters.
spectutils = data.spectutils

# Save vectors in a Dataset object.
vec_path = data_paths["vectors"]
print(f"Saving vector Dataset objects in folder {vec_path}")
data.clean_mixed_vectors_train_dataset.save(path=data_paths["vectors"]["train"])
data.clean_mixed_vectors_test_dataset.save(path=data_paths["vectors"]["test"])

# # Defines the transformations applied to a single tensor during preprocessing.
# def compose_preprocessing_steps(tens):
#     tens1 = tf.expand_dims(
#         spectutils.power_to_db(tf.abs(spectutils.tensor_spectrogram_from_tensor_audio(tens))),
#         axis=-1,
#     )
# 
#     if(not NORMALIZE):
#         return tens1
# 
#     min = tf.math.reduce_min(tens1)
#     max = tf.math.reduce_max(tens1)
#     shape = tf.shape(tens1)
#     ones = tf.ones(shape)
# 
#     if min == max:
#         return tf.zeros(shape)
# 
#     all_min = tf.math.scalar_mul(min, ones)
#     min_zero_tens = tf.math.subtract(tens1, all_min)
#     zero_one_tens = tf.math.scalar_mul(
#         tf.math.reciprocal(tf.math.subtract(max, min)), min_zero_tens
#     )
# 
#     all_zero_point_five = tf.math.scalar_mul(tf.constant(0.5), ones)
#     centered_tens = tf.math.subtract(zero_one_tens, all_zero_point_five)
#     normalized_tens = tf.math.scalar_mul(tf.constant(2.0), centered_tens)
# 
#     return normalized_tens
# 
# print("Mapping decibel spectrogram conversion over dataset... ")
# 
# # Dataset of training mixed-clean spectrogram pairs 
# train_spects: tf.data.Dataset = data.clean_mixed_vectors_train_dataset.map(
#     lambda tens1, tens2: (compose_preprocessing_steps(tens1), compose_preprocessing_steps(tens2)),
#     num_parallel_calls=tf.data.AUTOTUNE,
#     deterministic=False,
#     name="vecstospecttensors"
# )
# 
# # Dataset of testing mixed-clean spectrogram pairs
# test_spects: tf.data.Dataset = data.clean_mixed_vectors_test_dataset.map(
#     lambda tens1, tens2: (compose_preprocessing_steps(tens1), compose_preprocessing_steps(tens2)),
#     num_parallel_calls=tf.data.AUTOTUNE,
#     deterministic=False,
#     name="vecstospecttensors"
# )
# 
# # Notifies user of the first few sizes in the dataset
# print("Some training sizes: ")
# for spect1, spect2 in train_spects.take(10): # type: ignore
#     print("Mixed size: ", spect1.shape, "Clean size: ", spect2.shape)
# 
# print("Some testing sizes: ")
# for spect1, spect2 in test_spects.take(10): # type: ignore
#     print("Mixed size: ", spect1.shape, "Clean size: ", spect2.shape)
# 
# # Save spectrograms in Dataset objects.
# spect_path = data_paths["spectrograms"]
# print(f"Saving spectrogram Dataset objects in folder {spect_path}")
# train_spects.save(path=data_paths["spectrograms"]["train"])
# test_spects.save(path=data_paths["spectrograms"]["test"])
# 
# print("Done processing data!")

# For determining whether vectors are in correct mixed, clean order 
for vec1, vec2 in data.clean_mixed_vectors_train_dataset.take(1): # type: ignore
    data.spectutils.save_numpy_as_wav(vec1, "./first_example.wav")
    data.spectutils.save_numpy_as_wav(vec2, "./second_example.wav")
