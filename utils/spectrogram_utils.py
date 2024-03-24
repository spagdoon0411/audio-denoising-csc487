from random import randrange

import jaxtyping as jt
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import tensorflow as tf
from librosa import load
from matplotlib.axes import Axes
from matplotlib.figure import Figure

AudioVector = jt.Float[np.ndarray, "timesteps"]
SpectrogramMatrix = jt.Float[np.ndarray, "frequency timesteps"]

DEFAULT_NOISE_LEVEL = 0.1
USING_DB = False

# Provides functions useful for working with audio files and their spectrograms
class SpectUtils:
    def __init__(
        self, sampling_rate: int, hop_length: int, frame_length: int, fft_length: int
    ):
        self.sampling_rate: int = sampling_rate
        self.hop_length: int = hop_length
        self.frame_length = frame_length
        self.fft_length = fft_length

    # Loads a single audio file at the path into a floating-point Numpy "time series" vector, sampling at the provided
    # rate (samples per second)
    def load_into_numpy(self, path: str) -> AudioVector:
        audio_arr: AudioVector
        audio_arr, _ = load(path=path, sr=self.sampling_rate)

        return audio_arr

#     # Takes an AudioVector and returns a complex-valued frequency-wave vs. time SpectrogramMatrix
#     def spectrogram_from_numpy_audio(self, vec: AudioVector) -> SpectrogramMatrix:
#         stft_mat: SpectrogramMatrix = librosa.stft(
#             y=vec, hop_length=self.hop_length, center=True
#         )
#         # stft_mat: SpectrogramMatrix = librosa.feature.melspectrogram(y=vec, sr=self.sampling_rate, hop_length=self.hop_length)
#         # return np.abs(stft_vec)
#         return stft_mat

#     # Takes an AudioArray and returns an intensity-frequency vs. time SpectrogramMatrix (where intensity is
#     # measured in dB)
#     def decibel_spectrogram_from_numpy_audio(
#         self, vec: AudioVector
#     ) -> SpectrogramMatrix:
#         stft_mat: SpectrogramMatrix = self.spectrogram_from_numpy_audio(vec=vec)
#         return librosa.amplitude_to_db(
#             S=np.abs(stft_mat)
#         )  # TODO: this used to take a parameter ref=np.max
 
    # Takes an intensity spectrogram and provides a MatPlotLib figure that displays it with intensity-level
    # color coding. It is the user's responsibility to save or display the resulting figure.
    def display_intensity_spectrogram(
        self, spec: SpectrogramMatrix, title: str
    ) -> tuple[Figure, Axes]:
        # TODO: potentially dangerous type: ignore
        numpy_spec = spec if isinstance(spec, np.ndarray) else spec.numpy()  # type: ignore
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots()  # TODO: y_axis below used to be set to 'log'
        img = librosa.display.specshow(
            data=numpy_spec,
            y_axis=None,
            x_axis="time",
            ax=ax,
            hop_length=self.hop_length,
        )
        ax.set_title(label=title)
        fig.colorbar(mappable=img, ax=ax, format="%+2.0f dB")
        return fig, ax

#     # Takes an intensity (decibel) spectrogram and converts it to an audio vector, converting it to
#     # an amplitude spectrogram first.
#     def numpy_audio_from_db_spectrogram(self, spect: SpectrogramMatrix) -> AudioVector:
#         amplitude_spect: SpectrogramMatrix = librosa.db_to_amplitude(S_db=spect)
#         # amplitude_spect = spect
#         # TODO: does np.abs belong around this?
#         # return librosa.feature.inverse.mel_to_audio(M=amplitude_spect, sr=self.sampling_rate, hop_length=self.hop_length)
#         # return librosa.feature.inverse.mel_to_audio(M=amplitude_spect, sr=self.sampling_rate)
#         return librosa.griffinlim(
#             S=amplitude_spect, hop_length=self.hop_length, center=False
#         )

    # Stores numpy audio as a playable wav file
    def save_numpy_as_wav(self, vec: AudioVector, path: str) -> None:
        numpy_vec = vec if isinstance(vec, np.ndarray) else vec.numpy()  # type: ignore
        sf.write(file=path, data=numpy_vec, samplerate=self.sampling_rate)

#     # Composes intermediate functions above to obtain spectrograms from audio files. Optional image saving
#     # (not done if image_name is None).
#     def audio_to_db_spectrogram(
#         self,
#         path: str,
#         image_directory: None | str = None,
#         image_name: None | str = None,
#     ) -> SpectrogramMatrix:
#         spec: SpectrogramMatrix = self.decibel_spectrogram_from_numpy_audio(
#             vec=self.load_into_numpy(path=path)
#         )
#         if image_name != None:
#             if image_directory == None:
#                 image_directory = "."
# 
#             fig: Figure
#             _: Axes
#             fig, _ = self.display_intensity_spectrogram(
#                 spec=spec, title="Example Spectrogram"
#             )
#             # TODO: does this save to desired path?
#             fig.savefig(fname=path + "/" + image_name)
#         return spec

#     # Composes intermediate functions above to obtain audio files from spectrograms.
#     def db_spectrogram_to_audio(
#         self, directory: str, name: str, spect: SpectrogramMatrix
#     ) -> None:
#         self.save_numpy_as_wav(
#             vec=self.numpy_audio_from_db_spectrogram(spect=spect),
#             path=directory + "/" + name,
#         )

    # Takes a vector of clean audio, a vector of noise, and a noise level (0 to 100%, expressed as ratio) and
    # combines both. Optional randomstate.
    def clean_noise_mixer(
        self,
        cleanvector: AudioVector,
        noisevector: AudioVector,
        noise_level: float = DEFAULT_NOISE_LEVEL,
    ):
        sizedif: int = cleanvector.shape[0] - noisevector.shape[0]

        # Used in calculations for either case: size(noise) > size(clean) and size(clean) < size(noise)
        displacement: int = randrange(0, abs(sizedif) + 1)

        # Claim: the following calculations always create a noisy audio vector that is the same size as the
        # clean audio vector. Proof by cases:

        # Case 1: noise vectors have more components than clean audio vectors
        if sizedif < 0:
            # Trim the first 'displacement' components from the noise vector and the last 'abs(sizedif) - displacement' components.
            # Note that abs(sizedif) = -sizedif when sizedif < 0.
            noisevector = noisevector[
                displacement : noisevector.shape[0] - (-sizedif - displacement)
            ]

            # The resulting vector has:
            #
            # size(noise) - displacement - (abs(sizedif) - displacement)
            # size(noise) - displacement - abs(sizedif) + displacement
            # size(noise) - abs(sizedif)
            # size(noise) - (size(noise) - size(clean))
            # size(noise) - size(noise) + size(clean)
            # size(clean)
            #
            # ...components

            mixedvector = noise_level * noisevector + (1 - noise_level) * cleanvector

        # Case 2: noise vectors have fewer components than clean audio vectors
        else:
            # Prepend 'displacement' zeros and append abs(sizedif) - displacement zeros to the clean audio vector
            noisevector = np.concatenate(
                (
                    np.zeros((displacement,)),
                    noisevector,
                    np.zeros(sizedif - displacement),
                ),
                axis=0,
            )

            # The resulting vector has:
            #
            # size(noise) + displacement + abs(sizedif) - displacement
            # size(noise) + abs(sizedif)
            # size(noise) + sizedif
            # size(noise) + size(clean) - size(noise)
            # size(clean)
            #
            # ... components

            mixedvector = noise_level * noisevector + (1 - noise_level) * cleanvector

        return mixedvector

    def tensor_spectrogram_from_tensor_audio(self, tens):
        return tf.signal.stft(
            signals=tens,
            frame_length=self.frame_length,
            frame_step=self.hop_length,
            fft_length=self.fft_length,
            window_fn=tf.signal.hann_window,
        )

    def power_to_db(self, S, amin=1e-16, top_db=80.0):
        """Convert a power-spectrogram (magnitude squared) to decibel (dB) units.
        Computes the scaling ``10 * log10(S / max(S))`` in a numerically
        stable way.
        Based on:
        https://librosa.github.io/librosa/generated/librosa.core.power_to_db.html
        """

        if not USING_DB:
            return S

        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))  # type: ignore
            return numerator / denominator  # type: ignore

        # Scale magnitude relative to maximum value in S. Zeros in the output
        # correspond to positions where S == ref.
        ref = tf.reduce_max(S)

        log_spec = 10.0 * _tf_log10(tf.maximum(amin, S))
        log_spec -= 10.0 * _tf_log10(tf.maximum(amin, ref))

        log_spec = tf.maximum(log_spec, tf.reduce_max(log_spec) - top_db)

        return log_spec


