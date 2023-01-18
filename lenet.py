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
