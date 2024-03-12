import segmentation_models as sm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from segmentation_models import Unet
from keras.layers import Input, Conv2D
from keras.models import Model

base_model = Unet()
base_model = Unet(backbone_name='resnet101', encoder_weights='imagenet')

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
