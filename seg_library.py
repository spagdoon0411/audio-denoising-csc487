import segmentation_models.segmentation_models as sm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

from keras.layers import Input, Conv2D
from keras.models import Model
from data_paths import data_paths


base_model = sm.Unet()
base_model = sm.Unet(backbone_name='resnet101', encoder_weights='imagenet')

N = 1

inp = Input(shape=(None, None, N))
l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
out = base_model(l1)

model = Model(inp, out, name=base_model.name)
BACKBONE = 'resnet101'
preprocess_input = sm.get_preprocessing(BACKBONE)

# define model
model.compile(
'Adam',
loss = tf.keras.losses.MeanSquaredError(),
 metrics = ['mae']
)
# Obtain references to spectrogram datasets
train_data = tf.data.Dataset.load(data_paths["spectrograms"]["train"])
test_data = tf.data.Dataset.load(data_paths["spectrograms"]["test"])

# Obtain references to the vector datasets. TODO: remove 
test_vecs = tf.data.Dataset.load(data_paths["vectors"]["test"]).take(5)
train_vecs = tf.data.Dataset.load(data_paths["vectors"]["train"]).take(5)

weights_path = "."

# fitting model
checkpoint = ModelCheckpoint(weights_path+'/model_ResNet.h5', verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
history = model.fit(
    train_data,
    batch_size=1,
    epochs=1,
    validation_data=(test_data),
    callbacks=[checkpoint]
)
