from keras.api._v2.keras import Model
from keras.api._v2.keras.layers import (BatchNormalization, Conv2D,
                                        Conv2DTranspose, Dropout, Input,
                                        MaxPooling2D, ReLU, concatenate)

import tensorflow as tf

class OurUNet:
    # Takes a model specification dictionary that matches the UNet 
    # form (see training/model_spec.py for example)
    def build_model(self, modelspec: dict) -> Model:
        super().__init__()
        # Save model specification dictionary
        self.model_spec = modelspec

        # The input layer should accept images of variable size. 
        # The way to do this in Keras is to set dimensions of the 
        # shape to None.
        self.inputlayer = Input(shape=(None, None, 2), batch_size=None)

        # The most recent layer produced was the input layer. This 
        # reference is here for convenience (just like linked list 
        # references). Generally, this model IS built just like a linked
        # list.
        last_layer = self.inputlayer

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

        # Create the final output layer, feeding in the last upsampling layer as input.
        self.outputlayer = self.make_output_layer(
            input_layer=last_layer, output_spec=modelspec["output"]
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
            filters=2,#output_spec["num_classes"],
            kernel_size=output_spec["kernel_size"],
            activation=output_spec["activation"],
        )(input_layer)
