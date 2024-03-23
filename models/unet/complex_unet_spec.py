import json

model_spec = {
    "downsampling" : {
        "downsampling_1" : {
            "conv1" : {
                "filters" : 16 * 2,
                "kernel_size" : (3, 3),
                "activation" : "relu",
                "kernel_initializer" : "he_normal",
                "padding" : "same"
            },
            "dropout" : {
                "rate" : 0.1
            },
            "conv2" : {
                "filters" : 16 * 2,
                "kernel_size" : (3, 3),
                "activation" : "relu",
                "kernel_initializer" : "he_normal",
                "padding" : "same"
            },
            "max_pool" : {
                "pool_size" : (2, 2)
            }
        },
        "downsampling_2" : {
            "conv1" : {
                "filters" : 32 * 2,
                "kernel_size" : (3, 3),
                "activation" : "relu",
                "kernel_initializer" : "he_normal",
                "padding" : "same"
            },
            "dropout" : {
                "rate" : 0.1
            },
            "conv2" : {
                "filters" : 32 * 2,
                "kernel_size" : (3, 3),
                "activation" : "relu",
                "kernel_initializer" : "he_normal",
                "padding" : "same"
            },
            "max_pool" : {
                "pool_size" : (2, 2)
            }
        },
        "downsampling_3" : {
            "conv1" : {
                "filters" : 64 * 2,
                "kernel_size" : (3, 3),
                "activation" : "relu",
                "kernel_initializer" : "he_normal",
                "padding" : "same"
            },
            "dropout" : {
                "rate" : 0.2
            },
            "conv2" : {
                "filters" : 64 * 2,
                "kernel_size" : (3, 3),
                "activation" : "relu",
                "kernel_initializer" : "he_normal",
                "padding" : "same"
            },
            "max_pool" : {
                "pool_size" : (2, 2)
            }
        },
        "downsampling_4" : {
            "conv1" : {
                "filters" : 128 * 2,
                "kernel_size" : (3, 3),
                "activation" : "relu",
                "kernel_initializer" : "he_normal",
                "padding" : "same"
            },
            "dropout" : {
                "rate" : 0.2
            },
            "conv2" : {
                "filters" : 128 * 2,
                "kernel_size" : (3, 3),
                "activation" : "relu",
                "kernel_initializer" : "he_normal",
                "padding" : "same"
            },
            "max_pool" : {
                "pool_size" : (2, 2)
            }
        }
    },
    "valley" : {
        "conv1" : {
            "filters" : 256 * 2,
            "kernel_size" : (3, 3),
            "activation" : "relu",
            "kernel_initializer" : "he_normal",
            "padding" : "same"
        },
        "dropout" : {
            "rate" : 0.3
        },
        "conv2" : {
            "filters" : 256 * 2,
            "kernel_size" : (2, 2),
            "activation" : "relu",
            "kernel_initializer" : "he_normal",
            "padding" : "same"
        }
    },
    "upsampling" : {
        "upsampling1" : {
            "convt" : {
                "filters" : 128 * 2,
                "kernel_size" : (2,2),
                "strides" : (2, 2),
                "padding" : "same"
            }
        },
        "upsampling2" : { 
            "convt" : {
                "filters" : 64 * 2,
                "kernel_size" : (2,2),
                "strides" : (2, 2),
                "padding" : "same"
            }
        },
        "upsampling3" : { 
            "convt" : {
                "filters" : 32 * 2,
                "kernel_size" : (2,2),
                "strides" : (2, 2),
                "padding" : "same"
            }
        },
        "upsampling4" : { 
            "convt" : {
                "filters" : 16 * 2,
                "kernel_size" : (2,2),
                "strides" : (2, 2),
                "padding" : "same"
            }
        }
    },
    "output" : {
        "num_classes" : 2,
        "kernel_size" : (1, 1),
        "activation" : "sigmoid"
    }
}
print(json.dumps(model_spec,
                 sort_keys = False,
                 indent = 4))





