import os
import random

import tensorflow as tf
from spectrogram_utils import AudioVector, SpectUtils

ALLOWED_EXTS: list[str] = [".wav"]


# Provides functions for translating directories of audio files to data in formats
# that can be used for training models (namely, Dataset objects). Treated
# as a wrapper around two files that provides a multi-typed interface to the data.
class AudioData:

    def __init__(
        self,
        clean_audio_path: str,
        clean_audio_test_path: str,
        noisy_audio_path: str,
        noisy_audio_test_path: str,
        sampling_rate: int,
        hop_length: int,
        noise_level: float,
        clean_vec_repeats: int = 1,
    ):

        self.spectutils: SpectUtils = SpectUtils(
            sampling_rate=sampling_rate, hop_length=hop_length
        )

        print("Validating directories...")
        clean_audio_dir: bytes
        noisy_audio_dir: bytes
        clean_audio_dir, noisy_audio_dir = self.validate_directories(
            clean_audio_path, noisy_audio_path
        )

        clean_audio_test_dir: bytes
        noisy_audio_test_dir: bytes
        clean_audio_test_dir, noisy_audio_test_dir = self.validate_directories(
            clean_audio_test_path, noisy_audio_test_path
        )

        print("Loading clean training files as vectors...")
        self.clean_vectors, self.clean_names = self.files_to_vectors(
            clean_audio_path, clean_audio_dir
        )
        print("Loading noise training files as vectors...")
        self.noisy_vectors, self.noisy_names = self.files_to_vectors(
            noisy_audio_path, noisy_audio_dir
        )
        print("Loading clean testing files as vectors...")
        self.clean_test_vectors, self.clean_test_names = self.files_to_vectors(
            clean_audio_test_path, clean_audio_test_dir
        )
        print("Loading noise testing files as vectors...")
        self.noisy_test_vectors, self.noisy_test_names = self.files_to_vectors(
            noisy_audio_test_path, noisy_audio_test_dir
        )

        print("Converting clean and mixed training tensors to dataset... ")
        self.clean_mixed_vectors_train_dataset = (
            self.create_clean_mixed_dataset(
                self.clean_vectors, self.noisy_vectors, noise_level, clean_vec_repeats
            )
        )
        print("Converting clean and mixed testing tensors to dataset... ")
        self.clean_mixed_vectors_test_dataset = (
            self.create_clean_mixed_dataset(
                self.clean_test_vectors, self.noisy_test_vectors, noise_level, clean_vec_repeats
            )
        )

        # print("Performing train-test split... ")
        # self.train_dataset, self.test_dataset = tf.keras.utils.split_dataset(
        #     dataset=self.clean_mixed_vectors_dataset,
        #     left_size=train_proportion,
        #     shuffle=True,
        #     seed=random_seed,
        # )

        print("Done wrapping dataset!")


    def create_clean_mixed_dataset(
        self,
        clean_vectors: list[AudioVector],
        noise_vectors: list[AudioVector],
        noise_level,
        num_clean_repeats: int = 1,
    ):

        # The Tensor specification of an audio vector
        spect_spec = tf.TensorSpec(shape=(None,), dtype=tf.float32)

        # The Tensor specification of a mixed-and-label entry in the dataset
        spec = (spect_spec, spect_spec)

        generator = lambda: self.make_clean_mixed_generator(
            clean_vectors, noise_vectors, noise_level, num_clean_repeats
        )
        
        # Tensorflow datasets can be created from a generator that specifies 
        # how to create an entry. An output specification (corresponding to 
        # a single entry in the database) must be specified.
        return tf.data.Dataset.from_generator(generator, output_signature=spec)

    # Should not be used directly. Creates the generator that creates the dataset of mixed 
    # vectors and labels. 
    def make_clean_mixed_generator(
        self,
        clean_vectors: list[AudioVector],
        noise_vectors: list[AudioVector],
        noise_level,
        num_clean_repeats: int = 1,
    ):
        # Requesting more clean-vector repeats than noise vectors
        # is an error. 
        if(num_clean_repeats > len(noise_vectors)):
            raise Exception(f"Requested repeating each clean vector {num_clean_repeats} times, but there are only {len(noise_vectors)} noise vectors to use.")

        # This loop pairs a clean vector with `num_clean_repeats`, non-duplicate noise vectors.
        for clean in clean_vectors:
            # Sample as many noise vectors as there were repeats of clean vectors requested.
            for noise in random.sample(noise_vectors, num_clean_repeats):
                # Yield one clean-mixed pair per call. This increases the length of the
                # dataset without introducing structure.
                yield self.spectutils.clean_noise_mixer(
                    clean, noise, noise_level
                ), clean

    # Takes the clean and noisy audio paths and ensures they're (TODO: compatible)
    # directories of audio files
    def validate_directories(
        self, clean_audio_path: str, noisy_audio_path: str
    ) -> tuple[bytes, bytes]:

        # Ensures paths provided are actually directories
        if not os.path.isdir(clean_audio_path):
            raise (
                NotADirectoryError(
                    f"Clean audio path {clean_audio_path} is not a directory."
                )
            )

        if not os.path.isdir(noisy_audio_path):
            raise (
                NotADirectoryError(
                    f"Noisy audio path {noisy_audio_path} is not a directory."
                )
            )

        # Attempt to obtain references to directories
        noisy_audio_dir = os.fsencode(noisy_audio_path)
        clean_audio_dir = os.fsencode(clean_audio_path)

        return clean_audio_dir, noisy_audio_dir

    # Predicate for determining whether a file extension is supported by our models
    def file_extension_is_valid(self, file_name: str):
        ext: str
        _, ext = os.path.splitext(file_name)
        if ext not in ALLOWED_EXTS:
            print(
                f"File {file_name} is not one of supported types: {ALLOWED_EXTS}. Skipping."
            )
            return False

        return True

    # Maps a directory of audio files to a list of AudioVectors
    def files_to_vectors(
        self, dir_path: str, dir: bytes
    ) -> tuple[list[AudioVector], list[str]]:
        # Create a list of AudioVectors to add to as files get visited
        audiovecs: list[AudioVector] = []

        # A list of filenames to keep for convenience
        names: list[str] = []

        # Visit files in the provided directory one by one
        for fileentry in os.scandir(dir):
            # Ensure file extension is supported
            if self.file_extension_is_valid(fileentry.name.decode("utf-8")):
                # Convert file to AudioVector (Tensor) if it's valid and aggregate
                # with others
                vector: AudioVector = self.spectutils.load_into_numpy(
                    os.path.join(dir_path, fileentry.name.decode("utf-8"))
                )
                audiovecs.append(vector)
                # Record file name in the order it was visited
                names.append(fileentry.name.decode("utf-8"))
                # yield vector
            else:
                continue

        return audiovecs, names
