# About UNet Model

## Research
I adapted the UNet model definition to Keras library functions using this source's code: https://www.machinelearningnuggets.com/image-segmentation-with-u-net-define-u-net-model-from-scratch-in-keras-and-tensorflow/

This is the original UNet paper, in which it was proposed for use in biomedical image segmentation: https://arxiv.org/pdf/1505.04597.pdf?ref=machinelearningnuggets.com.

## To Do
3/3/24
[ ] Add layer names in layer definitions based on inputted specification so that model summaries and errors are informative
[ ] Bring repeated constants to the top of model_spec.py to allow for quick iteration
[ ] Double-check model_spec.py and unet.py for programming errors before compiling
[ ] Try compiling UNet and getting a graph representation through Keras utility function
[ ] Try training UNet on the tutorial's image segmentation dataset (above)
[ ] Make spectrograms out of entire dataset via function, and save as Tensor objects in a TensorFlow Dataset object

3/4/24
[ ] Write a dataset size/shape verifier function that takes a TensorFlow Dataset object and determines whether images are too small to apply all of the convolution operations proposed in a model specification
[ ] Try training UNet on a small set of spectrograms and verify that an audio file can be recovered
[ ] Write the data-label generation function (that makes linear combinations of clean and noisy audio)
[ ] Try training the UNet on a whole Dataset of clean-noisy spectrograms and verify that audio files can be recovered

3/5/24, 3/6/24
[ ] Wrap UNet in STFT and ISTFT layers to allow window size and window function type to become tunable hyperparameters
[ ] Try training the wrapped UNet on audio vectors (rather than spectrograms), to test wrapping. Verify that an audio file can be recovered.
[ ] Define a UNet hypermodel class (i.e., use Keras Tuner's Choice variables and related Keras hyperparameter classes to create an internal model specification dictionary)
[ ] Conduct UNet hypermodel training on a whole Dataset of audio vectors
[ ] Choose the best UNet hypermodel to train as a "final" product
