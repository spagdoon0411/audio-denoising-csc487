import tensorflow as tf
import os
from utils.spectrogram_utils import SpectUtils, AudioVector, SpectrogramMatrix

ALLOWED_EXTS : list[str] = [".wav"]

# Provides functions for translating directories of audio files to data in formats
# that can be used for training models (namely, Dataset objects). Treated
# as a wrapper around two files that provides a multi-typed interface to the data.
class AudioData:

    def __init__(self, clean_audio_path : str, noisy_audio_path : str, sampling_rate : int):
        self.spectutils : SpectUtils = SpectUtils(sampling_rate)

        clean_audio_dir : bytes
        noisy_audio_dir : bytes
        clean_audio_dir, noisy_audio_dir = self.validate_directories(clean_audio_path, noisy_audio_path)

        # For storing the names of files that this class wraps around. Useful for error
        # messages and later lookup by the user.
        self.clean_vectors, self.clean_names = self.files_to_vectors(clean_audio_path, clean_audio_dir)
        self.noisy_vectors, self.noisy_names = self.files_to_vectors(noisy_audio_path, noisy_audio_dir)

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
        if(ext not in ALLOWED_EXTS):
            raise Exception(f"File {file_name} is not one of supported types {ALLOWED_EXTS}")

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
            if(self.file_extension_is_valid(str(fileentry.name))):
                # Convert file to AudioVector (Tensor) if it's valid and aggregate with others
                vector : AudioVector = self.spectutils.load_into_numpy(os.path.join(dir_path, str(fileentry.name)))
                audiovecs.append(tf.convert_to_tensor(vector))

                # Record file name in the order it was visited
                names.append(str(fileentry.name))

        return audiovecs, names

    # Takes a list of AudioVectors and converts it to a list of SpectrogramMatrix objects
    def vectors_to_spectrograms(self, vectors : list[AudioVector]) -> list[SpectrogramMatrix]:
        # TODO: will fn_output_signature accept SpectrogramMatrix?
        return [self.spectutils.spectrogram_from_numpy_audio(vec) for vec in vectors]
