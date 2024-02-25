import matplotlib
from librosa import load, stft, display, amplitude_to_db
import librosa
import numpy as np
import nptyping as npt
from nptyping import NDArray, Shape, Float
from typing import Any
import matplotlib.pyplot as plt
from beartype import beartype

# Audio time series vectors are one-dimensional NumPy arrays.
AudioVector = NDArray[Shape["*"], Float]

# Spectrograms that result from applying STFT to AudioVectors are two-dimensional NumPy arrays 
SpectrogramMatrix = NDArray[Shape["*, *"], Float]

class SpectUtils:

    # Loads a single audio file at the path into a floating-point Numpy "time series" vector, sampling at the provided
    # rate (samples per second)
    def load_into_numpy(self, path : str, sample_rate : float) -> AudioVector:
        audio_arr : NDArray[Shape["*"], npt.Float] 
        returned_sample_rate : float
        audio_arr, returned_sample_rate = load(path=path, sr=sample_rate)

        print(f"Successfully loaded the audio file from {path} into a NumPy array, sampling at {sample_rate} samples/second")

        return audio_arr
    
    # Takes an AudioArray and returns an amplitude-frequency vs. time SpectrogramMatrix
    def spectrogram_from_numpy_audio(self, vec : AudioVector) -> SpectrogramMatrix:
        stft_vec: SpectrogramMatrix = stft(vec)
        return np.abs(stft_vec)
    
    # Takes an AudioArray and returns an intensity-frequency vs. time SpectrogramMatrix (where intensity is
    # measured in dB)
    def decibel_spectrogram_from_numpy_audio(self, vec : AudioVector) -> SpectrogramMatrix:
        stft_vec: SpectrogramMatrix = self.spectrogram_from_numpy_audio(vec)
        return librosa.amplitude_to_db(S=stft_vec, ref=np.max)
    
    # Takes an intensity spectrogram and provides a MatPlotLib figure that displays it with intensity-level
    # color coding. It is the user's responsibility to save or display the resulting figure.
    def display_intensity_spectrogram(self, spec: SpectrogramMatrix, title : str):
        fig, ax = plt.subplots()
        img = librosa.display.specshow(spec,y_axis='log', x_axis='time', ax=ax)
        ax.set_title(title)
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        return fig

s = SpectUtils()
audionp: AudioVector = s.load_into_numpy(path="./flickr_audio/wavs/667626_18933d713e_0.wav" , sample_rate=32000.0)
spec: SpectrogramMatrix = s.decibel_spectrogram_from_numpy_audio(vec=audionp)

print(spec.shape)

fig = s.display_intensity_spectrogram(spec=spec, title="Example Spectrogram")
fig.savefig("examplefromfunc")