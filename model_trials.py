from keras.models import Model
from keras.layers import Input , Conv2D
import inception

def trial_model(input_shape):

    X_input = Input(input_shape)
    X = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2')(X_input)
    X = inception.inception_module(X,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_1')

    X = inception.inception_module(X,
                         filters_1x1=64,
                         filters_3x3_reduce=96,
                         filters_3x3=128,
                         filters_5x5_reduce=16,
                         filters_5x5=32,
                         filters_pool_proj=32,
                         name='inception_2')

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model

model = trial_model(input_shape=(224, 224, 3))

model.summary()