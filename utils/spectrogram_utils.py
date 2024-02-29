from librosa import load
import librosa
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import jaxtyping as jt

AudioVector = jt.Float[np.ndarray, "timesteps"]
SpectrogramMatrix = jt.Float[np.ndarray, "frequency timesteps"]

class SpectUtils:

    def __init__(self, sampling_rate):
        self.sampling_rate: int = sampling_rate

    # Loads a single audio file at the path into a floating-point Numpy "time series" vector, sampling at the provided
    # rate (samples per second)
    def load_into_numpy(self, path : str) -> AudioVector:
        audio_arr : AudioVector
        returned_sample_rate : float
        audio_arr, returned_sample_rate = load(path=path, sr=self.sampling_rate)

        print(f"Successfully loaded the audio file from {path} into a NumPy array, sampling at {self.sampling_rate} samples/second")

        return audio_arr
    
    # Takes an AudioArray and returns an amplitude-frequency vs. time SpectrogramMatrix
    def spectrogram_from_numpy_audio(self, vec : AudioVector) -> SpectrogramMatrix:
        stft_vec: SpectrogramMatrix = librosa.feature.melspectrogram(y=vec, sr=self.sampling_rate)
        return np.abs(stft_vec) 
    
    # Takes an AudioArray and returns an intensity-frequency vs. time SpectrogramMatrix (where intensity is
    # measured in dB)
    def decibel_spectrogram_from_numpy_audio(self, vec : AudioVector) -> SpectrogramMatrix:
        stft_vec: SpectrogramMatrix = self.spectrogram_from_numpy_audio(vec=vec)
        return librosa.amplitude_to_db(S=stft_vec, ref=np.max)
    
    # Takes an intensity spectrogram and provides a MatPlotLib figure that displays it with intensity-level
    # color coding. It is the user's responsibility to save or display the resulting figure.
    def display_intensity_spectrogram(self, spec: SpectrogramMatrix, title : str) -> tuple[Figure, Axes]:
        fig: Figure 
        ax: Axes
        fig, ax = plt.subplots()
        img = librosa.display.specshow(data=spec,y_axis='log', x_axis='time', ax=ax)
        ax.set_title(label=title)
        fig.colorbar(mappable=img, ax=ax, format="%+2.0f dB")
        return fig, ax
    
    # Takes an intensity (decibel) spectrogram and converts it to an audio vector, converting it to
    # an amplitude spectrogram first.
    def numpy_audio_from_decibel_spectrogram(self, spect : SpectrogramMatrix) -> AudioVector:
        amplitude_spect : SpectrogramMatrix = librosa.db_to_amplitude(S_db=spect)
        # TODO: does np.abs belong around this?
        return np.abs(librosa.feature.inverse.mel_to_audio(M=amplitude_spect, sr=self.sampling_rate))
    
    # Stores numpy audio as a playable wav file
    def save_numpy_as_wav(self, vec : AudioVector, path : str) -> None:
        # TODO: this write method has a subtype argument. What should I use?
        sf.write(file=path, data=vec, samplerate=self.sampling_rate)

    # Composes intermediate functions above to obtain spectrograms from audio files. Optional image saving
    # (not done if image_name is None).
    def audio_to_spectrogram(self, path : str, image_directory : str, image_name : None | str = None) -> SpectrogramMatrix:
        spec : SpectrogramMatrix = self.decibel_spectrogram_from_numpy_audio(vec=self.load_into_numpy(path=path))
        if(image_name != None):
            if(image_directory == None):
                image_directory = "."

            fig : Figure
            ax : Axes
            fig, ax = self.display_intensity_spectrogram(spec=spec, title="Example Spectrogram")
            # TODO: does this save to desired path?
            fig.savefig(fname=path + "/" + image_name)
        return spec
    
    # Composes intermediate functions above to obtain audio files from spectrograms.
    def spectrogram_to_audio(self, directory : str, name : str, spect : SpectrogramMatrix) -> None:
        self.save_numpy_as_wav(vec=self.numpy_audio_from_decibel_spectrogram(spect=spect), path=directory + "/" + name)
    

# s = SpectUtils(sampling_rate=64000)
# audionp: AudioVector = s.load_into_numpy(path="./data/flickr_audio/wavs/667626_18933d713e_0.wav")
# spec: SpectrogramMatrix = s.decibel_spectrogram_from_numpy_audio(vec=audionp)

# print(spec.shape)

# fig : Figure
# ax : Axes
# fig, ax = s.display_intensity_spectrogram(spec=spec, title="Example Spectrogram")
# fig.savefig(fname="examplefromfunc2")

# # TODO: how does the sound of this audio vector compare to the old one? Why are there negative values??
# reconstructed_vec: AudioVector = s.numpy_audio_from_decibel_spectrogram(spect=spec)
# print(reconstructed_vec.shape)
# print(audionp.shape)

# s.save_numpy_as_wav(vec=reconstructed_vec, path="./test.wav")