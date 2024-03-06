from models.unet.unet import OurUNet
from first_unet_spec import model_spec

unet : OurUNet = OurUNet(modelspec=model_spec)
