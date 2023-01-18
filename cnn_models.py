from tensorflow import keras

def Conv(X, conv_feature_maps=4, conv_kernel=(3, 3), conv_strides=(1, 1),
         conv_padding='same', activation='relu',
         name=None):
    """
    Help function for convolutional + max pooling layers

    Arguments:
    X -- input tensor

    Returns:
    model -- a Model() instance in TensorFlow
    """
    # CONV -> Batch Normalization -> ReLU Block applied to X (3 lines of code)
    X = keras.layers.Conv2D(conv_feature_maps, conv_kernel,
                            strides=conv_strides, padding=conv_padding, activation=None, name=name)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation(activation)(X)

    return X


def ConvPool(X, conv_feature_maps=4, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', activation='relu',
             pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same', name=None):
    """
    Help function for convolutional + max pooling layers

    Arguments:
    X -- input tensor

    Returns:
    model -- a Model() instance in TensorFlow
    """

    # CONV -> Batch Normalization -> ReLU Block applied to X (3 lines of code)
    X = keras.layers.Conv2D(conv_feature_maps, conv_kernel,
                            strides=conv_strides, padding=conv_padding, activation=None, name=name)(X)
    X = keras.layers.BatchNormalization(axis=-1)(X)
    X = keras.layers.Activation(activation)(X)

    # MAX-POOL (1 line of code)
    X = keras.layers.MaxPooling2D(
        pool_size, pool_strides, padding=pool_padding)(X)

    return X
