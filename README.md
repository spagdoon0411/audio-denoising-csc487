# audio-denoising-csc487
## Collaborators

- Daniel Casares-Iglesias
- Grant Baer
- Spandan Suthar

## Tasks: 

- [X] Create tools for converting audio files to spectrograms (as real-valued 2D arrays), using `librosa`
- [X] Create tools for visualizing spectrograms 
- [X] Create tools for converting spectrograms back to audio files, also using `librosa`
- [ ] Determine what exactly a model should take as input and provide as output
    - `SpectrogramMatrix` -> `SpectrogramMatrix` representing the noise mask? 
    - `SpectrogramMatrix` -> `SpectrogramMatrix` representing the clean audio?
- [ ] Convert dataset to spectrograms (to avoid converting each file repeatedly while training)
- [ ] Program a UNet model that maps signal + noise SpectrogramMatrix objects to clean SpectrogramMatrix objects
- [ ] Train the UNet model above 

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

> Do note that [Beartype's annotations seem to have issues with NumPy (at least, I don't understand how they interact)](https://github.com/beartype/beartype/issues/334), so you may have to forego type hints on functions involving NumPy types. - Spandan 

We're dealing with NumPy arrays during preprocessing, etc. Specifically, time series audio vectors and spectrograms resulting from short-time Fourier transforms are defined as the following:

```python3 
AudioVector = NDArray[Shape["*"], Float]
SpectrogramMatrix = NDArray[Shape["*, *"], Float]
```

