from models.unet.unet import OurUNet
from models.unet.test_unet_spec import model_spec
import tensorflow as tf
from utils.spectrogram_utils import SpectUtils

test_data = tf.data.Dataset.load("/Volumes/baseqi1tb/ms_dataset_spects/test")
train_data = tf.data.Dataset.load("/Volumes/baseqi1tb/ms_dataset_spects/train")
for thing1, thing2 in train_data.take(10): # type: ignore
    print("mixed shape: ", thing1.shape)
    print("clean shape: ", thing2.shape)

for thing1, thing2 in test_data.take(10): # type: ignore
    print("mixed shape: ", thing1.shape)
    print("clean shape: ", thing2.shape)


spectutils = SpectUtils(sampling_rate=16000, hop_length=256)

test_vecs = tf.data.Dataset.load("/Volumes/baseqi1tb/ms_dataset_vectors/test")
train_vecs = tf.data.Dataset.load("/Volumes/baseqi1tb/ms_dataset_vectors/train")

for mixedvec, cleanvec in train_vecs.take(1):
    spectutils.save_numpy_as_wav(mixedvec, "./example_mixed.wav")
    spectutils.save_numpy_as_wav(cleanvec, "./example_clean.wav")

test_data=test_data.batch(5)
train_data=train_data.batch(5)

unetbuilder = OurUNet()
unet = unetbuilder.build_model(modelspec=model_spec)
unet.compile(optimizer='adam', loss="mse", metrics=["accuracy"])
unet.fit(train_data, validation_data=test_data, epochs=1, batch_size=5, shuffle=True)

