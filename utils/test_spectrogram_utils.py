from spectrogram_utils import SpectUtils, AudioVector, SpectrogramMatrix
import matplotlib.pyplot as plt
import librosa

s = SpectUtils(sampling_rate=22050, hop_length=512)

test_clean_path = "./data/MS-SNSD/clean_test/clnsp0.wav"
test_noise_path = "./data/MS-SNSD/noise_test/Babble_1.wav"

# NOTE: this should be called from the root directory of the project (i.e., the one that CONTAINS data)

test_clean_vec: AudioVector = s.load_into_numpy(test_clean_path)
test_noise_vec: AudioVector = s.load_into_numpy(test_noise_path)
print("Dimensions of clean vector before operations: ", test_clean_vec.shape)
print("Dimensions of noise vector before operations: ", test_noise_vec.shape)

comparison_spec: SpectrogramMatrix = s.decibel_spectrogram_from_numpy_audio(test_clean_vec)
print("Dimensions of spectrogram directly from clean audio: ", comparison_spec.shape)
print("Doing linear combination operation...")

mixed_vec: AudioVector = s.clean_noise_mixer(cleanvector=test_clean_vec, noisevector=test_noise_vec, noise_level=0.1)
print("Dimensions of mixed audio vector after operations: ", mixed_vec.shape)

print("Attempting to make mixed spectrogram...")
mixed_spec: SpectrogramMatrix = s.decibel_spectrogram_from_numpy_audio(vec=mixed_vec)

fig1, ax1 = s.display_intensity_spectrogram(spec=comparison_spec, title="Example Spectrogram")
fig1.savefig(fname="comparison_spec")

fig2, ax2 = s.display_intensity_spectrogram(spec=mixed_spec, title="Example Spectrogram After Audio Mixing")
fig2.savefig(fname="mixed_spec")

noise_spec = s.decibel_spectrogram_from_numpy_audio(vec=test_noise_vec)
fig3, ax3 = s.display_intensity_spectrogram(spec=noise_spec, title="Spectrogram Noise Vector Used in Mixing")
fig3.savefig(fname="noise_spec")

s.save_numpy_as_wav(test_clean_vec, path="./test_clean_vec.wav")
s.save_numpy_as_wav(test_noise_vec, path="./test_noise_vec.wav")
s.save_numpy_as_wav(mixed_vec, path="./mixed_vec.wav")

s.db_spectrogram_to_audio(directory=".", name="test_clean_vec_recovered.wav", spect=comparison_spec)

testspect : SpectrogramMatrix = s.audio_to_db_spectrogram(librosa.ex('trumpet'))
fig4, ax4 = s.display_intensity_spectrogram(spec=testspect, title="See if empty spect")

plt.show()

# TODO: how does the sound of this audio vector compare to the old one? Why are there negative values??
# reconstructed_vec: AudioVector = s.numpy_audio_from_decibel_spectrogram(spect=spec)
# print(reconstructed_vec.shape)
# print(audionp.shape)

# s.save_numpy_as_wav(vec=reconstructed_vec, path="./test.wav")
