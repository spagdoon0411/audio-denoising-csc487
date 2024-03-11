from models.unet.unet import OurUNet
from models.unet.test_unet_spec import model_spec
import tensorflow as tf
from utils.spectrogram_utils import SpectUtils
from data_paths import data_paths 

# Obtain references to spectrogram datasets
train_data = tf.data.Dataset.load(data_paths["spectrograms"]["train"])
test_data = tf.data.Dataset.load(data_paths["spectrograms"]["test"])

spectutils = SpectUtils(sampling_rate=sample_rate, hop_length=hop_length)

test_vecs = tf.data.Dataset.load(data_paths["vectors"]["test"])
train_vecs = tf.data.Dataset.load(data_paths["vectors"]["train"])

for mixedvec, cleanvec in train_vecs.take(1):
    spectutils.save_numpy_as_wav(mixedvec, "./example_mixed.wav")
    spectutils.save_numpy_as_wav(cleanvec, "./example_clean.wav")

test_data=test_data.batch(5)
train_data=train_data.batch(5)

unetbuilder = OurUNet()
unet = unetbuilder.build_model(modelspec=model_spec)
unet.compile(optimizer='adam', loss="mse", metrics=["accuracy"])
unet.fit(train_data, validation_data=test_data, epochs=1, batch_size=5, shuffle=True)

