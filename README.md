# audio-denoising-csc487
## Collaborators

- Daniel Casares-Iglesias
- Grant Baer
- Spandan Suthar

## Tasks: 

- [X] Create tools for converting audio files to spectrograms (as real-valued 2D arrays), using `librosa`
- [X] Create tools for visualizing spectrograms 
- [ ] Create tools for converting spectrograms back to audio files, also using `librosa`
- [ ] Determine what exactly a model should take as input and provide as output
    - SpectrogramMatrix -> SpectrogramMatrix representing the noise mask? 
    - SpectrogramMatrix -> SpectrogramMatrix representing the clean audio?


## Documentation

### Types
> I'm using PyLint + type hinting to enforce some static typing and to make types visible. I'd *highly* recommend Beartype, too. - Spandan 

> Do note that [Beartype's annotations seem to have issues with NumPy (at least, I don't understand how they interact)](https://github.com/beartype/beartype/issues/334), so you may have to forego type hints on functions involving NumPy types. - Spandan 

We're dealing with NumPy arrays during preprocessing, etc. Specifically, time series audio vectors and spectrograms resulting from short-time Fourier transforms are defined as the following:

```python3 
AudioVector = NDArray[Shape["*"], Float]
SpectrogramMatrix = NDArray[Shape["*, *"], Float]
```

