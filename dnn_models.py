# Create Models
from tensorflow import keras
import tensorflow as tf


# FUNCTION: DNN_Model
def DNN_Model(input_shape1):
    """
    Implementation of the DNN_Model

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in TensorFlow
    """
    # Define the input placeholder as a tensor with shape input_shape.
    X_input = keras.Input(input_shape1)

    Y_input = keras.Input(1)

    # FLATTEN THE TENSOR
    X = keras.layers.Flatten()(X_input)

    X = keras.layers.Dropout(0.5)(X)

    X = keras.layers.Dense(128, activation='relu')(X)
    X = keras.layers.Dropout(0.5)(X)

    Y = keras.layers.Dense(32, activation='relu')(Y_input)

    Z = keras.layers.concatenate(values=[X,Y])


    Z = keras.layers.Dense(64, activation='relu')(Z)
    Z = keras.layers.Dropout(0.5)(Z)

    Z = keras.layers.Dense(32, activation='relu')(Z)
    Z = keras.layers.Dropout(0.5)(Z)

    output = keras.layers.Dense(1)(Z)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = keras.Model(inputs=[X_input,Y_input], outputs=output, name='DNN_Model1')

    return model
