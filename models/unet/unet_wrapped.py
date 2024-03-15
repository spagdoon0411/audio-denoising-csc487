from re import I
import tensorflow as tf
from tensorflow._api.v2.signal import stft
import tensorflow_io as tfio
from keras.api._v2.keras import Model
from keras.api._v2.keras.layers import (BatchNormalization, Conv2D,
                                        Conv2DTranspose, Dropout, Input,
                                        MaxPooling2D, ReLU, concatenate)

# Borrowed from https://github.com/timsainb/tensorflow2-generative-models/blob/master/7.0-Tensorflow-spectrograms-and-inversion.ipynb

def _tf_log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype)) # type: ignore # TODO
    return numerator / denominator # type: ignore # TODO

def _amp_to_db_tensorflow(x):
    return 20 * _tf_log10(tf.clip_by_value(tf.abs(x), 1e-5, 1e100))

def _db_to_amp_tensorflow(x):
    return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

class STFTLayer(tf.keras.layers.Layer):
    def __init__(self, stft_config):
        self.stft_config = stft_config

    def call(self, inputs, *args, **kwargs):
        super.__call__(args, kwargs) # TODO: correct?

        output = tf.signal.stft(
            inputs,
            frame_length=self.stft_config["frame_length"],
            frame_step=self.stft_config["hop_length"],
            fft_length=self.stft_config["fft_length"],
            window_fn=self.stft_config["window_func"]
        )

        return output


class ISTFTLayer(tf.keras.layers.Layer):
    def __init__(self, stft_config):
        self.stft_config = stft_config
    
    def call(self, inputs, *args, **kwargs):
        super.__call__(args, kwargs) # TODO: correct?

        output = tf.signal.inverse_stft(
            inputs,
            frame_length=self.stft_config["frame_length"],
            frame_step=self.stft_config["hop_length"],
            fft_length=self.stft_config["fft_length"],
            window_fn=self.stft_config["window_func"]
        )

        return output

class AmpDbSpectLayer(tf.keras.layers.Layer):
    def __init__(self, stft_config, dB = False):
        self.stft_config = stft_config
        self.dB = dB
    
    def call(self, inputs, *args, **kwargs):
        super.__call__(args, kwargs) # TODO: correct?

        output = tf.signal.stft(
            inputs,
            frame_length=self.stft_config["frame_length"],
            frame_step=self.stft_config["hop_length"],
            fft_length=self.stft_config["fft_length"],
            window_fn=self.stft_config["window_func"]
        )

        output = tf.math.abs(output)

        if(self.dB):
            output = _amp_to_db_tensorflow(output)

        return output

class AmpDbToAudioLayer(tf.keras.layers.Layer):
    def __init__(self, stft_config, dB = False, halve = False, iters = 30):
        self.stft_config = stft_config
        self.dB = dB
        self.iters = iters
        self.halve = halve
    
    def call(self, inputs, *args, **kwargs):
        super.__call__(args, kwargs) # TODO: correct?

        if(self.dB):
            inputs = _db_to_amp_tensorflow(inputs)
        
        if(self.halve):
            inputs = 0.5 * inputs

        # TODO: remove 0.5?
        # TODO: params could be different
        output = tfio.audio.inverse_spectrogram(
            spectrogram=inputs,
            nfft=self.stft_config["fft_length"],
            window=self.stft_config["frame_length"],
            stride=self.stft_config["hop_length"],
            iterations=self.iters
        )

        return output

# type is either "AMP" or "COMP"

class OurUNet:
    # Takes a model specification dictionary that matches the UNet 
    # form (see training/model_spec.py for example)
    def build_model(self, modelspec: dict, stft_config, type="AMP") -> Model:
        super().__init__()

        stft_config = {
            "hop_length" : 256,
            "noise_level" : 0.1,
            "frame_length" : 256 * 4,
            "fft_length" : 256 * 4,
            "window_func" : tf.signal.hann_window
        }

        # Save model specification dictionary
        self.model_spec = modelspec

        # The input layer should accept vectors of variable length. 
        # The way to do this in Keras is to set dimensions of the 
        # shape to None.
        self.inputlayer = Input(shape=(None, ), batch_size=None)

        # The most recent layer produced was the input layer. This 
        # reference is here for convenience (just like linked list 
        # references). Generally, this model IS built just like a linked
        # list.
        last_layer = self.inputlayer

        # Choose the method for mapping audio vectors to spectrograms
        # based on the model type. AMP uses STFT -> amplitude -> dB 
        # conversion; COMP uses raw STFT, multiple channels.
        if(type == "AMP"):
            last_layer = AmpDbSpectLayer(
                stft_config=stft_config, 
                dB=True
            )
        elif(type == "COMP"):
            last_layer = STFTLayer(stft_config=stft_config)
        else:
            raise Exception(f"{type} is not a valid model type. Choose from AMP or COMP.")

        # Stores convolutional layers whose outputs will be used as future layer
        # inputs. These are the references that will be "passed across the U" so 
        # that filters resulting from convolutional layers during the descent of 
        # the U can be used as inputs to U-ascent convolutional layers (via 
        # keras.layers.concatenate).
        self.second_conv_layers = []

        # Descend the "U": produce downsampling layers
        for downsample_spec in modelspec["downsampling"].values():
            # The relevant references needed from a downsampling block are the 
            # second convolutional layer (to allow filters to be passed across 
            # the U) and the final layer in the block (the pooling layer)
            second_conv_layer, pooling_layer = self.make_downsampling_layer(
                input_layer=last_layer, downsample_spec=downsample_spec
            )

            # Save the second convolutional layer from the downsampling 
            # layer just created; it feeds into later upsampling layers.
            self.second_conv_layers.append(second_conv_layer)

            # The last layer created (the one whose outputs will feed directly 
            # into the next layer in the U itself) was the pooling layer.
            last_layer = pooling_layer

        # Create the layer at the bottom of the U
        valley = self.make_valley_layer(last_layer, modelspec["valley"])
        last_layer = valley

        # Layers are concatenated in the upsampling layers in the reverse 
        # order relative to how they were created (i.e., in stack order). 
        # Doing this odd reverse here makes later iteration
        # easier.
        self.second_conv_layers.reverse()

        # Ascend the U: produce upsampling layers which take filters from 
        # the previous convolutional layer and filters from
        # corresponding convolutional layers in the U's descent.
        for downsample_conv_layer, upsample_spec in zip(
            self.second_conv_layers, modelspec["upsampling"].values()
        ):
            # The important reference to keep here is just the final layer 
            # of the new upsampling layer, to feed into
            # the next upsampling layer.
            last_layer = self.make_upsampling_layer(
                input_layer=last_layer,
                downsample_conv_layer=downsample_conv_layer,
                upsample_spec=upsample_spec,
            )

        # The image output layer of the UNet feeds into an ISTFT
        # or Griffin-Lim layer to audio vectors.
        last_layer = self.make_output_layer(
            input_layer=last_layer, output_spec=modelspec["output"]
        )

        # Choose the method for mapping spectrograms back to audio based
        # on the model type. AMP uses Griffin-Lim, COMP uses ISTFT.
        if(type == "AMP"):
            self.outputlayer = AmpDbToAudioLayer(
                stft_config=stft_config, 
                dB=True,
                halve=False,
                iters=30
            )
        elif(type == "COMP"):
            self.outputlayer = ISTFTLayer(
                stft_config=stft_config
            )

        return Model(inputs=[self.inputlayer], outputs=[self.outputlayer])

    # Takes a specification for a downsampling layer and returns a reference
    # to the second convolutional layer and a reference to the final pooling
    # layer (to be given as an argument to the next layer--downsampling or
    # valley). Also takes an input layer.
    def make_downsampling_layer(self, input_layer, downsample_spec: dict):
        conv1spec: dict = downsample_spec["conv1"]
        dropoutspec: dict = downsample_spec["dropout"]
        conv2spec: dict = downsample_spec["conv2"]
        poolspec: dict = downsample_spec["max_pool"]

        conv1 = Conv2D(
            filters=conv1spec["filters"],
            kernel_size=conv1spec["kernel_size"],
            activation=conv1spec["activation"],
            kernel_initializer=conv1spec["kernel_initializer"],
            padding=conv1spec["padding"],
        )(input_layer)

        dropout = Dropout(rate=dropoutspec["rate"])(conv1)

        conv2 = Conv2D(
            filters=conv2spec["filters"],
            kernel_size=conv2spec["kernel_size"],
            activation=conv2spec["activation"],
            kernel_initializer=conv2spec["kernel_initializer"],
            padding=conv2spec["padding"],
        )(dropout)

        norm = BatchNormalization()(conv2)

        relu = ReLU()(norm)

        pooling = MaxPooling2D(pool_size=poolspec["pool_size"])(relu)

        # Return BOTH the second convolutional layer and the pooling
        # layer, as both are inputs for future layers.
        return conv2, pooling

    # Takes a specification for the "valley" layer (the bottommost UNet layer)
    # and creates it, using the provided input layer and returning the last layer
    # for the caller to feed into future layers.
    def make_valley_layer(self, input_layer, valley_spec: dict):
        conv1spec = valley_spec["conv1"]
        dropoutspec = valley_spec["dropout"]
        conv2spec = valley_spec["conv2"]

        conv1 = Conv2D(
            filters=conv1spec["filters"],
            kernel_size=conv1spec["kernel_size"],
            activation=conv1spec["activation"],
            kernel_initializer=conv1spec["kernel_initializer"],
            padding=conv1spec["padding"],
        )(input_layer)

        norm = BatchNormalization()(conv1)

        relu = ReLU()(norm)

        dropout = Dropout(rate=dropoutspec["rate"])(relu)

        conv2 = Conv2D(
            filters=conv2spec["filters"],
            kernel_size=conv2spec["kernel_size"],
            activation=conv2spec["activation"],
            kernel_initializer=conv2spec["kernel_initializer"],
            padding=conv2spec["padding"],
        )(dropout)

        return conv2

    # Takes a specification for an upsampling layer and creates it, using the 
    # provided input layer and the provided convolutional layer.
    def make_upsampling_layer(
        self, input_layer, downsample_conv_layer, upsample_spec: dict
    ):

        convtspec = upsample_spec["convt"]

        convt = Conv2DTranspose(
            filters=convtspec["filters"],
            kernel_size=convtspec["kernel_size"],
            strides=convtspec["strides"],
            padding=convtspec["padding"],
        )(input_layer)

        resized_convt_layer = tf.image.resize(images = convt,
                                             size = tf.shape(downsample_conv_layer)[1:3],
                                             method=tf.image.ResizeMethod.BILINEAR, 
                                             preserve_aspect_ratio=False)
        concat = concatenate([resized_convt_layer, downsample_conv_layer])

       #  resized_downsample_conv_layer = tf.image.resize(images = downsample_conv_layer,
       #                                                  size = tf.shape(convt)[1:3],
       #                                                  method=tf.image.ResizeMethod.BILINEAR, 
       #                                                  preserve_aspect_ratio=False)


        # concat = concatenate([convt, resized_downsample_conv_layer])

        norm = BatchNormalization()(concat)

        relu = ReLU()(norm)

        return relu

    # Produces a final convolutional output layer, using the provided input layer.
    def make_output_layer(self, input_layer, output_spec):
        # The output layer is a single convolutional layer.
        return Conv2D(
            filters=output_spec["num_classes"],
            kernel_size=output_spec["kernel_size"],
            activation=output_spec["activation"],
        )(input_layer)
