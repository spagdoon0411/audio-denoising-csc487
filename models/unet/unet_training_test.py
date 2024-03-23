from unet import OurUNet
from test_unet_spec import model_spec
from keras.api._v2.keras.utils import plot_model
from keras.api._v2.keras.callbacks import EarlyStopping, TensorBoard

import tensorflow as tf

# The OurUNet initializer simply takes a model specification.
unet_builder : OurUNet = OurUNet()
unet = unet_builder.build_model(modelspec=model_spec)
unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
plot_model(unet, "first_unet.png")

callbacks = [EarlyStopping(patience=2, monitor='val_loss'),
             TensorBoard(log_dir='logs')]

train_save_path = "data/unettest/train"
test_save_path = "data/unettest/test"

testset = tf.data.Dataset.load(test_save_path).batch(batch_size = 1, drop_remainder=True)
trainset = tf.data.Dataset.load(test_save_path).batch(batch_size = 1, drop_remainder=True)

unet.fit(trainset, validation_data=testset, batch_size=16, epochs=25, callbacks=callbacks)
