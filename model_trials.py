from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, AveragePooling2D, Dense, Dropout, Flatten
from inception import inception_module
from resnet import bottleneck_residual_block


def trial_model(input_shape):
    X_input = Input(input_shape,name='Input')

    # first stage
    X = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X_input)
    X = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)

    # second stage
    X = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
    X = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)

    # second stage
    X = inception_module(X,
                         filters_1x1=256,
                         filters_3x3_reduce=64,
                         filters_3x3=256,
                         filters_5x5_reduce=16,
                         filters_5x5=64,
                         filters_pool_proj=64,
                         name='inception_1')

    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)

    # third stage
    X = bottleneck_residual_block(X, 3, [64, 64, 256], stage=3, block='a', reduce=True, s=1)
    X = bottleneck_residual_block(X, 3, [64, 64, 256], stage=3, block='b')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(X)

    # fourth stage
    X = bottleneck_residual_block(X, 3, [64, 64, 256], stage=4, block='a', reduce=True, s=1)
    X = bottleneck_residual_block(X, 3, [64, 64, 256], stage=4, block='b')

    X = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid')(X)

    X = Flatten()(X)
    X = Dropout(0.5)(X)
    X = Dense(1000, activation='relu')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


model = trial_model(input_shape=(224, 224, 3))

num_params = model.count_params()
print(f'Number of parameters: {num_params:,}\n')

model.summary()
