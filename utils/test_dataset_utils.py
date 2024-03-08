import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from dataset_utils import AudioData
from spectrogram_utils import AudioVector

if __name__ == "__main__":
    clean_audio_path = "./data/MS-SNSD/clean_train"
    clean_audio_test_path = "./data/MS-SNSD/clean_test"
    noise_audio_path = "./data/MS-SNSD/noise_train"
    noise_audio_test_path = "./data/MS-SNSD/noise_test"

    data: AudioData = AudioData(
        clean_audio_path=clean_audio_path,
        noisy_audio_path=noise_audio_path,
        clean_audio_test_path=clean_audio_test_path,
        noisy_audio_test_path=noise_audio_test_path,
        sampling_rate=22050,
        hop_length=512,
        noise_level=0.1,
        clean_vec_repeats=1
    )

    print("Grabbing vectors from dataset: ")
    test_dataset = data.clean_mixed_vectors_train_dataset.take(5)
    figs = []
    for vec_pair in test_dataset:
        mixed_vec: AudioVector
        clean_vec: AudioVector

        mixed_tens, clean_tens = vec_pair # type: ignore

        mixed_vec = mixed_tens.numpy()
        clean_vec = clean_tens.numpy()

        print("Mixed vector: ", mixed_vec)
        print("Clean vector: ", clean_vec)

    def tensor_spectrogram_from_tensor_audio(tens):
        return tf.signal.stft(
            signals=tens,
            frame_length=512 * 4,
            frame_step=512,
            fft_length=512 * 4,
            window_fn=tf.signal.hann_window,
        )

    def power_to_db(S, amin=1e-16, top_db=80.0):
        """Convert a power-spectrogram (magnitude squared) to decibel (dB) units.
        Computes the scaling ``10 * log10(S / max(S))`` in a numerically
        stable way.
        Based on:
        https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
        """

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

    print("Mapping spectrogram conversion over dataset... ")

    # Attempt to map a spectrogram conversion over a few vectors in the dataset
    testspecs: tf.data.Dataset = data.clean_mixed_vectors_train_dataset.map(
        lambda tens1, tens2: (
            power_to_db(tf.abs(tensor_spectrogram_from_tensor_audio(tens1))),
            power_to_db(tf.abs(tensor_spectrogram_from_tensor_audio(tens2))),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
        name="vecstospecttensors"
    )

    stats = tfds.benchmark(testspecs)
    print(stats.stats)

    # Display the first spectrogram in the dataset
    figs = []
    for mixed_spect, clean_spect in testspecs.take(1):  # type: ignore
        fig, _ = data.spectutils.display_intensity_spectrogram(
            mixed_spect, "Mixed Spectrogram from Dataset"
        )

        fig2, _ = data.spectutils.display_intensity_spectrogram(
            clean_spect, "Clean Spectrogram from Dataset"
        )

        figs.append(fig)
    
    plt.show()
