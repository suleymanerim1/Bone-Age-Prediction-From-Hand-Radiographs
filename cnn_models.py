import tensorflow as tf
from tensorflow import keras


# LeNet Model
def LeNet_Model(input_shape):
    # Define input tensor with specified shape
    X_input = keras.Input(input_shape)

    # Add convolutional layer with 6 filters, kernel size of (5, 5), stride of (1, 1), and hyperbolic tangent activation
    X = keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(
        1, 1), activation='tanh', input_shape=input_shape, padding='same')(X_input)

    # Add average pooling layer with pool size of (2, 2) and stride of 2
    X = keras.layers.AveragePooling2D(
        pool_size=(2, 2), strides=2, padding='valid')(X)

    # Add convolutional layer with 16 filters, kernel size of (5, 5), stride of (1, 1), and hyperbolic tangent
    # activation
    X = keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(
        1, 1), activation='tanh', padding='valid')(X)

    # Add average pooling layer with pool size of (2, 2) and stride of 2
    X = keras.layers.AveragePooling2D(
        pool_size=(2, 2), strides=2, padding='valid')(X)

    # Add convolutional layer with 120 filters, kernel size of (5, 5), stride of (1, 1), and hyperbolic tangent
    # activation
    X = keras.layers.Conv2D(120, kernel_size=(5, 5), strides=(
        1, 1), activation='tanh', padding='valid')(X)

    # Flatten tensor
    X = keras.layers.Flatten()(X)

    Y_input = keras.Input(1)
    Y = keras.layers.Dense(32, activation='relu')(Y_input)

    Z = keras.layers.concatenate(inputs=[X, Y])

    # Add fully-connected dense layer with 84 units and hyperbolic tangent activation
    Z = keras.layers.Dense(84, activation='tanh')(Z)

    # Add final dense layer with single unit and no activation function
    Z = keras.layers.Dense(1)(Z)

    # Create model instance with input tensor and output tensor
    model = keras.Model(inputs=[X_input, Y_input], outputs=Z, name='leNet')

    return model


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

    # MAXPOOL (1 line of code)
    X = keras.layers.MaxPooling2D(
        pool_size, pool_strides, padding=pool_padding)(X)

    return X


def CNN_Model(input_shape):
    """
    Implementation of the CNN Model

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in TensorFlow
    """

    # START CODE HERE ### (1 line of code for each instruction)
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = keras.Input(input_shape)

    # FIRST CONV + MAXPOOL BLOCK
    X = ConvPool(X_input, conv_feature_maps=32, conv_kernel=(11, 11), conv_strides=(1, 1),
                 conv_padding='same', activation='relu',
                 pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same', name="Conv1")

    X = ConvPool(X, conv_feature_maps=64, conv_kernel=(5, 5), conv_strides=(1, 1),
                 conv_padding='same', activation='relu',
                 pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same', name="Conv2")

    X = ConvPool(X, conv_feature_maps=128, conv_kernel=(5, 5), conv_strides=(1, 1),
                 conv_padding='same', activation='relu',
                 pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same', name="Conv3")

    X = ConvPool(X, conv_feature_maps=256, conv_kernel=(5, 5), conv_strides=(1, 1),
                 conv_padding='same', activation='relu',
                 pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same', name="Conv4")

    # FLATTEN THE TENSOR
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dropout(0.4)(X)

    X = keras.layers.Dense(256, activation='relu', name="Dense1")(X)
    X = keras.layers.Dropout(0.4)(X)
    X = keras.layers.Dense(128, activation='relu', name="Dense2")(X)
    X = keras.layers.Dropout(0.4)(X)
    X = keras.layers.Dense(1, activation=None, name="Output_Layer")(X)

    # Create model
    model = keras.Model(inputs=X_input, outputs=X, name='CNN_Model')

    return model


def AlexNet_Model(input_shape):
    X_input = keras.Input(input_shape)

    # 1st layer (conv + pool + batchnorm)
    X = keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid',
                            kernel_regularizer=keras.regularizers.l2(0.0005))(X_input)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(X)
    X = keras.layers.BatchNormalization()(X)

    # 2nd layer (conv + pool + batchnorm)
    X = keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same',
                            kernel_regularizer=keras.regularizers.l2(0.0005))(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(X)
    X = keras.layers.BatchNormalization()(X)

    # layer 3 (conv + batchnorm)      <--- note that the authors did not add a POOL layer here
    X = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same',
                            kernel_regularizer=keras.regularizers.l2(0.0005))(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.BatchNormalization()(X)

    # layer 4 (conv + batchnorm)      <--- similar to layer 3
    X = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same',
                            kernel_regularizer=keras.regularizers.l2(0.0005))(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.BatchNormalization()(X)

    # layer 5 (conv + batchnorm)
    X = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',
                            kernel_regularizer=keras.regularizers.l2(0.0005))(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(X)

    # Flatten the CNN output to feed it with fully connected layers
    X = keras.layers.Flatten()(X)

    # Gender information layer 1
    Y_input = keras.Input(1)
    Y = keras.layers.Dense(32, activation='relu')(Y_input)

    # Concatenate Image features with gender features
    Z = keras.layers.concatenate(inputs=[X, Y])

    # layer 6 (Dense layer + dropout)
    Z = keras.layers.Dense(4096, activation='relu')(Z)
    Z = keras.layers.Dropout(0.5)(Z)

    # layer 7 (Dense layer + dropout)
    Z = keras.layers.Dense(4096, activation='relu')(Z)
    Z = keras.layers.Dropout(0.5)(Z)

    # layer 8 (softmax output layer)
    output = keras.layers.Dense(1)(Z)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = keras.Model(inputs=[X_input, Y_input], outputs=output, name='AlexNet')

    return model
