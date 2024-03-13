import tensorflow as tf

from data_paths import data_paths
from models.unet.test_unet_spec import model_spec
from models.unet.unet import OurUNet
from data_paths import data_config
from utils.spectrogram_utils import SpectUtils
from keras.api._v2.keras.utils import plot_model
import sys
import pickle


spectutils = SpectUtils(sampling_rate=data_config["sample_rate"],
                        hop_length=data_config["hop_length"],
                        frame_length=data_config["frame_length"],
                        fft_length=data_config["fft_length"])


earlystop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=1,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,
)

history = tf.keras.callbacks.History()

model_path = sys.argv[2]
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Obtain references to spectrogram datasets
train_data = tf.data.Dataset.load(data_paths["spectrograms"]["train"])
test_data = tf.data.Dataset.load(data_paths["spectrograms"]["test"])

# Obtain references to the vector datasets. TODO: remove 
test_vecs = tf.data.Dataset.load(data_paths["vectors"]["test"])
train_vecs = tf.data.Dataset.load(data_paths["vectors"]["train"])

for mixedvec, cleanvec in train_vecs.take(1):
    spectutils.save_numpy_as_wav(mixedvec, "./example_mixed.wav")
    spectutils.save_numpy_as_wav(cleanvec, "./example_clean.wav")

test_data = test_data.batch(1)
train_data = train_data.batch(1)

epochs = int(sys.argv[3])
unetbuilder = OurUNet()
unet = unetbuilder.build_model(modelspec=model_spec)
unet.compile(optimizer="adam", loss="mse", metrics=["mse", "mae", "val_mse", "val_mae"])
unet.fit(train_data, validation_data=test_data, epochs=epochs, batch_size=1, shuffle=True, callbacks=[earlystop, history, cp_callback])

history_path = sys.argv[1]
with open(history_path, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
