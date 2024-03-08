# audio-denoising-csc487
## Collaborators

- Daniel Casares-Iglesias
- Grant Baer
- Spandan Suthar

## Tasks: 

- [X] Create tools for converting audio files to spectrograms (as real-valued 2D arrays), using `librosa`
- [X] Create tools for visualizing spectrograms 
- [X] Create tools for converting spectrograms back to audio files, also using `librosa`
- [X] Program a UNet model that reads from a specification dictionary
- [X] Test aforementioned UNet model on simple test data (likely fixed dimensions)
- [X] Convert a signal dataset and a noise dataset into testing and training Tensorflow Dataset objects that can be stored in files
    - [X] Write a function that takes a clean AudioVector and a noise AudioVector and performs a linear combination according to a provided noise ratio (between 0 and 1)
    - [X] Write a function that takes a directory of clean .wav files and pure noise .wav files and generates a Tensorflow Dataset object where each member is a tuple: the first tuple member is a noisy AudioVector (created using the function above) and the second tuple member is the clean AudioVector.
    - [X] Write a function that maps the AudioVector to SpectrogramMatrix conversion over the whole dataset aforementioned.
    - [X] Save the Dataset resulting from the process above to a file (Tensorflow stores file fragments in a directory when you save a dataset with .save. .load just takes this directory of fragments.)
    - [X] Write a final function that maps a directory of clean .wav files and a directory of pure noise .wav files to a tuple of Datasets (one training and one test), then stores both datsets in directories with Tensorflow's .save function.
        - Noise ratio to use for each sample
        - The train-test split ratio
        - The name of the directory to save the testing set at
        - The name of the directory to save the training set at
        - A path to a directory in which the user stores their data
- [ ] Try training the UNet model with the spectrogram datasets resulting from the process in the previous task
- [ ] Adjust UNet model as necessary to get it to train
- [ ] Do some ad hoc tuning to get reasonable loss metrics
- [ ] Set up a UNet hypermodel and train via KerasTuner
- [ ] Do additional research to determine how to improve the UNet model. What have other people using UNet for audio denoising done?

## Documentation and Standards

### .gitignore
Add any datasets to your `.gitignore` file! They're usually too large to package with any code that people are downloading and versioning. A great `.gitignore` file is simply:

```
/data/*
```

### File Organization
Any new models should go into the models directory. Add a description, related research, and your name to the `model_descriptions.md` file. All models should be defined as a class, ideally a TensorFlow [Model](https://www.tensorflow.org/guide/keras/making_new_layers_and_models_via_subclassing) class (or a descendent).

### Types
> I'm using PyLint + type hinting to enforce some static typing and to make types visible. Jaxtyping provides type annotations on NumPy and Tensorflow objects. - Spandan 

> Do note that [Beartype's annotations seem to have issues with NumPy's typing package (at least, I don't understand how they interact)](https://github.com/beartype/beartype/issues/334). Jaxtyping is therefore more ideal. - Spandan 

We're dealing with NumPy arrays during preprocessing, etc. Specifically, time series audio vectors and spectrograms resulting from short-time Fourier transforms are defined as the following:

```python3 
AudioVector = jt.Float[np.ndarray, "timesteps"]
SpectrogramMatrix = jt.Float[np.ndarray, "frequency timesteps"]
```

