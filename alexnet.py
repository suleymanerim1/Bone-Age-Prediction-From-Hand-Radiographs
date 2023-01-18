from tensorflow import keras

def AlexNet_Model(input_shape):
    X_input = keras.Input(input_shape)

    # 1st layer (conv + pool + batch-norm)
    X = keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid',
                            kernel_regularizer=keras.regularizers.l2(0.0005))(X_input)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(X)
    X = keras.layers.BatchNormalization()(X)

    # 2nd layer (conv + pool + batch-norm)
    X = keras.layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same',
                            kernel_regularizer=keras.regularizers.l2(0.0005))(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(X)
    X = keras.layers.BatchNormalization()(X)

    # layer 3 (conv + batch-norm)      <--- note that the authors did not add a POOL layer here
    X = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same',
                            kernel_regularizer=keras.regularizers.l2(0.0005))(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.BatchNormalization()(X)

    # layer 4 (conv + batch-norm)      <--- similar to layer 3
    X = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same',
                            kernel_regularizer=keras.regularizers.l2(0.0005))(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.BatchNormalization()(X)

    # layer 5 (conv + batch-norm)
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
