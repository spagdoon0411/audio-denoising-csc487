from models.unet.unet import OurUNet
from first_unet_spec import model_spec
from keras.api._v2.keras.utils import plot_model

# The OurUNet initializer simply takes a model specification.
unet_builder : OurUNet = OurUNet()
unet = unet_builder.build_model(modelspec=model_spec)
unet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
plot_model(unet, "first_unet.png")

