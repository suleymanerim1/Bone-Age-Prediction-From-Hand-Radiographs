# Create Models
from tensorflow import keras
import tensorflow as tf


# FUNCTION: DNN_Model
def DNN_Model(input_shape):
    """
    Implementation of the DNN_Model

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in TensorFlow
    """
    # Define the input placeholder as a tensor with shape input_shape.
    X_input = keras.Input(input_shape)

    # FLATTEN THE TENSOR
    X = keras.layers.Flatten()(X_input)

    X = keras.layers.Dense(64, activation='relu')(X)
    X = keras.layers.Dropout(0.5)(X)

    X = keras.layers.Dense(32, activation='relu')(X)
   #X = keras.layers.Dropout(0.5)(X)

    output = keras.layers.Dense(1)(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = keras.Model(inputs=X_input ,outputs=output, name='DNN_Model1')

    return model


#model = DNN_Model(input_shape=(224, 224, 3))
#num_params = model.count_params()
#print(f'Number of parameters: {num_params:,}\n')
#model.summary()