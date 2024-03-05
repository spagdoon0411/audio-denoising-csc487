from keras.api._v2.keras.layers import Input, Conv2D, Dropout, BatchNormalization, ReLU, MaxPooling2D, Conv2DTranspose, concatenate

class OurUNet:
    # Takes a model specification dictionary that matches the UNet form (see model_spec.py for 
    # example)
    def __init__(self, modelspec : dict):

        self.model_spec = modelspec

        self.inputlayer = Input()

        # Stores convolutional layers whose outputs will be used as future layer
        # inputs.
        self.second_conv_layers = []
        
        # The most recent layer produced was the input layer
        last_layer = self.inputlayer
        
        # Descend the "U": produce downsampling layers
        for downsample_spec in modelspec["downsampling"]:
            second_conv_layer, pooling_layer = self.make_downsampling_layer(input_layer = last_layer,
                                                                            downsample_spec = downsample_spec)
            # Save the second convolutional layer from the downsampling layer just created;
            # it feeds into later upsampling layers.
            self.second_conv_layers.append(second_conv_layer)

            # The last layer created was the pooling layer.
            last_layer = pooling_layer

        # Create the layer at the "bottom" of the "U"
        valley = self.make_valley_layer(last_layer, modelspec["valley"])
        last_layer = valley

        # Ascend the "U": produce upsampling layers which also take corresponding
        # downsampling layers' convolutional ouputs
        for downsample_conv_layer, upsample_spec in zip(self.second_conv_layers, modelspec["upsampling"]):
            last_layer = self.make_upsampling_layer(input_layer = last_layer,
                                               downsample_conv_layer = downsample_conv_layer,
                                               upsample_spec = upsample_spec)

        # Once upsampling is completed, the model's work is done. The last upsampling layer
        # (specifically, the last layer within the last set of upsampling layers) IS this 
        # model's output layer.
        self.outputlayer = last_layer
    
    # Takes a specification for a downsampling layer and returns a reference
    # to the second convolutional layer and a reference to the final pooling
    # layer (to be given as an argument to the next layer--downsampling or
    # valley). Also takes an input layer.
    def make_downsampling_layer(self, input_layer, downsample_spec : dict):
        conv1spec = downsample_spec["conv1"]
        dropoutspec = downsample_spec["dropout"]
        conv2spec = downsample_spec["conv2"]
        poolspec = downsample_spec["max_pool"]

        conv1 = Conv2D(filters = conv1spec["filters"],
                       kernel_size = conv1spec["kernel_size"],
                       activation = conv1spec["activation"],
                       kernel_initializer = conv1spec["kernel_initializer"],
                       padding = conv1spec["padding"])(input_layer)
        
        dropout = Dropout(rate = dropoutspec["rate"])(conv1)

        conv2 = Conv2D(filters = conv2spec["filters"],
                       kernel_size = conv2spec["kernel_size"],
                       activation = conv2spec["activation"],
                       kernel_initializer = conv2spec["kernel_initializer"],
                       padding = conv2spec["padding"])(dropout)

        norm = BatchNormalization()(conv2)

        relu = ReLU()(norm)

        pooling = MaxPooling2D(pool_size = poolspec["pool_size"])(relu)
            
        # Return BOTH the second convolutional layer and the pooling
        # layer, as both are inputs for future layers.
        return conv2, pooling

    # Takes a specification for the "valley" layer (the bottommost UNet layer)
    # and creates it, using the provided input layer and returning the last layer
    # for the caller to feed into future layers.
    def make_valley_layer(self, input_layer, valley_spec : dict):
        conv1spec = valley_spec["conv1"]
        dropoutspec = valley_spec["dropout"]
        conv2spec = valley_spec["conv2"]

        conv1 = Conv2D(filters = conv1spec["filters"],
                       kernel_size = conv1spec["kernel_size"],
                       activation = conv1spec["activation"],
                       kernel_initializer = conv1spec["kernel_initializer"],
                       padding = conv1spec["padding"])(input_layer)

        norm = BatchNormalization()(conv1)

        relu = ReLU()(norm)

        dropout = Dropout(rate = dropoutspec["rate"])(relu)

        conv2 = Conv2D(filters = conv2spec["filters"],
                       kernel_size = conv2spec["kernel_size"],
                       activation = conv2spec["activation"],
                       kernel_initializer = conv2spec["kernel_initializer"],
                       padding = conv2spec["padding"])(dropout)

        return conv2
    
    # Takes a specification for an upsampling layer and creates it, using the provided
    # input layer and the provided convolutional layer.
    def make_upsampling_layer(self, input_layer, downsample_conv_layer, upsample_spec : dict):
        convtspec = upsample_spec["convt"]

        convt = Conv2DTranspose(filters = convtspec["filters"],
                                kernel_size = convtspec["kernel_size"],
                                strides = convtspec["strides"],
                                padding = convtspec["padding"])(input_layer)

        concat = concatenate([convt, downsample_conv_layer])

        norm = BatchNormalization()(concat)

        relu = ReLU()(norm)

        return relu

