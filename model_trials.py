from keras.layers import Input, Conv2D, MaxPool2D, AveragePooling2D, Dense, Dropout, Flatten, BatchNormalization, \
    Activation, Add
from keras.models import Model
from keras.regularizers import L2
from tensorflow.python.ops.init_ops_v2 import glorot_uniform

from inception import inception_module_without_bn_without_weightdecay
from resnet import bottleneck_residual_block_without_bn_without_weightdecay

reg = L2(0.0001)


def model_no_dropout_skip_bn_weightdecay(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = Activation('relu')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2')(X)
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool1')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='conv3')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='conv4')(X)
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool2')(X)

    # second stage
    X = inception_module_without_bn_without_weightdecay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='inception_1')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool3')(X)

    # third stage
    X = inception_module_without_bn_without_weightdecay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='inception_2')

    # forth stage
    X = bottleneck_residual_block_without_bn_without_weightdecay(X, 3, [128, 128, 256], stage=4, block='a', reduce=True,
                                                                 s=2)
    X = bottleneck_residual_block_without_bn_without_weightdecay(X, 3, [128, 128, 256], stage=4, block='b')

    X = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid', name='average_pool')(X)

    X = Flatten()(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


def trial_model(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(0.0001),
               name='conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='batch_norm1')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=L2(0.0001),
               name='conv2')(X)
    X = BatchNormalization(axis=3, name='batch_norm2')(X)
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool1')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=L2(0.0001), padding='same',
               name='conv3')(X)
    X = BatchNormalization(axis=3, name='batch_norm3')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), kernel_regularizer=L2(0.0001), padding='same',
               name='conv4')(X)
    X = BatchNormalization(axis=3, name='batch_norm4')(X)
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool2')(X)

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    X_shortcut = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=L2(0.0001))(
        X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)
    # second stage
    X = inception_module_without_bn_without_weightdecay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='inception_1')

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='max_pool3')(X)

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    X_shortcut = Conv2D(filters=352, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_regularizer=L2(0.0001))(
        X_shortcut)
    X_shortcut = BatchNormalization(axis=3)(X_shortcut)
    # third stage
    X = inception_module_without_bn_without_weightdecay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='inception_2')
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    # forth stage
    X = bottleneck_residual_block_without_bn_without_weightdecay(X, 3, [128, 128, 256], stage=4, block='a', reduce=True,
                                                                 s=2)
    X = bottleneck_residual_block_without_bn_without_weightdecay(X, 3, [128, 128, 256], stage=4, block='b')

    X = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid', name='average_pool')(X)

    X = Flatten()(X)
    X = Dropout(0.5, name='dropout-1')(X)
    X = Dense(1000, activation='relu', kernel_regularizer=L2(0.0001), name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dropout(0.5, name='dropout-2')(X)
    X = Dense(256, activation='relu', kernel_regularizer=L2(0.0001), name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


model = model_no_dropout_skip_bn_weightdecay(input_shape=(224, 224, 3))

num_params = model.count_params()
print(f'Number of parameters: {num_params:,}\n')
#model.summary()
