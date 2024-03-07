import librosa 
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

sampling_rate = 16000
hop_length = 512
example_audio_path = "./data/MS-SNSD/clean_test/clnsp0.wav"
example_audio, sr = librosa.load(example_audio_path, sr=sampling_rate)
stft_mat = librosa.stft(example_audio)
stft_mat_db = librosa.amplitude_to_db(np.abs(stft_mat))

# Takes an intensity spectrogram and provides a MatPlotLib figure that displays it with intensity-level
# color coding. It is the user's responsibility to save or display the resulting figure.
def display_intensity_spectrogram(spec, title) -> tuple:
    fig, ax = plt.subplots() # TODO: y_axis below used to be set to 'log'
    img = librosa.display.specshow(data=spec,y_axis=None, x_axis='time', ax=ax, hop_length=hop_length)
    ax.set_title(label=title)
    fig.colorbar(mappable=img, ax=ax, format="%+2.0f dB")
    return fig, ax

vec_recovered = librosa.griffinlim(S=librosa.db_to_amplitude(stft_mat_db), hop_length=hop_length)

# Stores numpy audio as a playable wav file
def save_numpy_as_wav(vec, path : str) -> None:
    # TODO: this write method has a subtype argument. What should I use?
    sf.write(file=path, data=vec, samplerate=sampling_rate)

save_numpy_as_wav(vec_recovered, "./test_trumpet_recovered.wav")
save_numpy_as_wav(example_audio, "./test_trumpet.wav")

fig, ax = display_intensity_spectrogram(stft_mat_db, title="test")

plt.show()
