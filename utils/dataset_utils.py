import os
from spectrogram_utils import SpectUtils, AudioVector, SpectrogramMatrix
import random
from itertools import chain

ALLOWED_EXTS : list[str] = [".wav"]

# Provides functions for translating directories of audio files to data in formats
# that can be used for training models (namely, Dataset objects). Treated
# as a wrapper around two files that provides a multi-typed interface to the data.
class AudioData:

    def __init__(self, clean_audio_path : str, noisy_audio_path : str, sampling_rate : int, hop_length : int, noise_level : float, clean_vec_repeats : int = 1):
        self.spectutils : SpectUtils = SpectUtils(sampling_rate=sampling_rate, hop_length=hop_length)

        print("Validating directories...")
        clean_audio_dir : bytes
        noisy_audio_dir : bytes
        clean_audio_dir, noisy_audio_dir = self.validate_directories(clean_audio_path, noisy_audio_path)

        print("Loading clean files as vectors...")
        self.clean_vectors, self.clean_names = self.files_to_vectors(clean_audio_path, clean_audio_dir)
        print("Loading noise files as vectors...")
        self.noisy_vectors, self.noisy_names = self.files_to_vectors(noisy_audio_path, noisy_audio_dir)

        print("Mixing vectors...")
        self.mixed_vectors = self.mix_clean_and_noisy(self.clean_vectors, self.noisy_vectors, noise_level, clean_vec_repeats)
        
        print("Creating clean spectrograms...")
        self.clean_spectrograms = self.vectors_to_spectrograms(self.clean_vectors)
        print("Creating mixed spectrograms...")
        self.mixed_spectrograms = self.vectors_to_spectrograms(self.mixed_vectors)

        print("Done wrapping dataset!")

    # Takes a list of AudioVectors and returns a Tensorflow Dataset of spectogram Tensors. A Dataset is used because
    # a Dataset is an iterator 
    def vectors_to_spectrograms_in_batches(self):
        return None

    # Takes the clean and noisy audio paths and ensures they're (TODO: compatible) directories
    # of audio files
    def validate_directories(self, clean_audio_path : str, noisy_audio_path : str) -> tuple[bytes, bytes]:
        
        # Ensures paths provided are actually directories
        if(not os.path.isdir(clean_audio_path)):
            raise(NotADirectoryError(f"Clean audio path {clean_audio_path} is not a directory."))
        
        if(not os.path.isdir(noisy_audio_path)):
            raise(NotADirectoryError(f"Noisy audio path {noisy_audio_path} is not a directory."))

        # Attempt to obtain references to directories
        noisy_audio_dir = os.fsencode(noisy_audio_path)
        clean_audio_dir = os.fsencode(clean_audio_path)

        return clean_audio_dir, noisy_audio_dir 

    # Predicate for determining whether a file extension is supported by our models
    def file_extension_is_valid(self, file_name:str):
        ext : str
        _, ext = os.path.splitext(file_name)
        # print(ext)
        # print(file_name)
        if(ext not in ALLOWED_EXTS):
            print(f"File {file_name} is not one of supported types: {ALLOWED_EXTS}. Skipping.")
            return False

        return True

    # Maps a directory of audio files to a list of AudioVectors
    def files_to_vectors(self, dir_path : str, dir : bytes) -> tuple[list[AudioVector], list[str]]:
        # Create a list of AudioVectors to add to as files get visited
        audiovecs : list[AudioVector] = []

        # A list of filenames to keep for convenience
        names : list[str] = []

        # Visit files in the provided directory one by one
        for fileentry in os.scandir(dir):
            # Ensure file extension is supported
            if(self.file_extension_is_valid(fileentry.name.decode("utf-8"))):
                # Convert file to AudioVector (Tensor) if it's valid and aggregate with others
                vector : AudioVector = self.spectutils.load_into_numpy(os.path.join(dir_path, fileentry.name.decode("utf-8")))
                audiovecs.append(vector)

                # Record file name in the order it was visited
                names.append(fileentry.name.decode("utf-8"))
            else:
                continue

        return audiovecs, names

    def convert_vectors_to_spectrograms(self, vectors : list[AudioVector]) -> list[SpectrogramMatrix]:
        return [self.spectutils.spectrogram_from_numpy_audio(vec) for vec in vectors]


    # Randomly mixes a list of clean AudioVectors and a list of noise AudioVectors to produce a single
    # list of AudioVectors. Uses the provided noise_level, a float between 0 and 1 representing the
    # coefficient of the noisy vector in the mixture linear combination. The resulting list has exactly
    def mix_clean_and_noisy(self, clean_vectors : list[AudioVector], noise_vectors : list[AudioVector], noise_level : float, repeat_clean : int = 1):

        # Takes a clean vector, pairs it with a random selection of `repeat_clean` noise vectors, and performs the linear
        # combination for each to generate a list of mixed vectors.
        def get_mixtures_for_clean(clean : AudioVector):
            return [self.spectutils.clean_noise_mixer(clean, noise, noise_level) for noise in random.sample(noise_vectors, repeat_clean)] 
       
        # Flatten the list before returning (because Python does not allow spread operators in list comprehensions)
        return list(chain.from_iterable([get_mixtures_for_clean(clean) for clean in clean_vectors]))

    # Takes a list of AudioVectors and converts it to a list of SpectrogramMatrix objects
    def vectors_to_spectrograms(self, vectors : list[AudioVector]) -> list[SpectrogramMatrix]:
        # TODO: will fn_output_signature accept SpectrogramMatrix?
        return [self.spectutils.spectrogram_from_numpy_audio(vec) for vec in vectors]


