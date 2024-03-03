import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa.feature.inverse


class AudioProcessor:
    def __init__(self, dataset_path, audio_output_path, spectrogram_output_path, target_length):
        self.dataset_path = dataset_path
        self.audio_output_path = audio_output_path
        self.spectrogram_output_path = spectrogram_output_path
        self.target_length = target_length

        # Dictionary to store mapping between spectrogram files and their original audio files
        self.spectrogram_to_audio_mapping = {}


        # create dirs if needed
        os.makedirs(audio_output_path, exist_ok=True)
        os.makedirs(spectrogram_output_path, exist_ok=True)

    def process_dataset(self):
        for filename in os.listdir(self.dataset_path):
            if filename.endswith('.wav'):
                #print(filename)
                file_path = os.path.join(self.dataset_path, filename)
                self.process_audio(file_path)

    def process_audio(self, file_path):
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        #print("Sample rate of the original audio:", sr)

        # Trim/pad audio to the target length
        if len(y) > self.target_length:
            y = y[:self.target_length]
        else:
            y = np.pad(y, (0, max(0, self.target_length - len(y))), 'constant')

        # Save processed audio
        output_audio_path = os.path.join(self.audio_output_path, os.path.basename(file_path))
        wavfile.write(output_audio_path, sr, y)

        # Store mapping between spectrogram and audio file
        spectrogram_filename = os.path.basename(output_audio_path).replace('.wav', '.npy')
        self.spectrogram_to_audio_mapping[spectrogram_filename] = os.path.basename(file_path)

        # Generate and save spectrogram
        self.generate_spectrogram(y, sr, output_audio_path)

    def generate_spectrogram(self, y, sr, audio_path):
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # spectrogram as a NumPy array
        spectrogram_filename = os.path.basename(audio_path).replace('.wav', '.npy')
        spectrogram_path = os.path.join(self.spectrogram_output_path, spectrogram_filename)
        np.save(spectrogram_path, S_dB)

        # save spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.tight_layout()
        plt.savefig(spectrogram_path.replace('.npy', '.png'))
        plt.close()


    def spectrogram_to_audio(self, spectrogram_path, output_audio_path):
        # Get the corresponding audio file for the spectrogram
        audio_filename = self.spectrogram_to_audio_mapping[os.path.basename(spectrogram_path)]

        # Get the sample rate of the original audio
        audio_file_path = os.path.join(self.dataset_path, audio_filename)
        sr = get_sample_rate(audio_file_path)
        print("Sample rate of the original audio:", sr)

        # Load the spectrogram
        S_dB = np.load(spectrogram_path)
        S = librosa.db_to_power(S_dB)

        # Inverse transform the Mel spectrogram to audio using the correct sample rate
        y_reconstructed = librosa.feature.inverse.mel_to_audio(S, sr=sr)

        # Save the reconstructed audio with the correct sample rate
        wavfile.write(output_audio_path, sr, y_reconstructed)


def get_sample_rate(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    return sr


# example usage
""" 
# defining paths for input, and two output files
dataset_path = '../audio_denoising/dataset'
audio_output_path = '../audio_denoising/outputs/audio_out'
spectrogram_output_path = '../audio_denoising/outputs/spectrogram'
target_length = 22050 * 5  # 1 second at 22050 Hz

# for creating spectrograms from audio
processor = AudioProcessor(dataset_path, audio_output_path, spectrogram_output_path, target_length)
processor.process_dataset()

# defining paths for sprectogram input, and audio output file
spectrogram_path = '../audio_denoising/outputs/spectrogram/taunt.npy'
output_audio_path = '../audio_denoising/outputs/reconstructed_audio/taunt_reconstructed.wav'

# for creating audio from spectrograms
processor.spectrogram_to_audio(spectrogram_path, output_audio_path)
### """