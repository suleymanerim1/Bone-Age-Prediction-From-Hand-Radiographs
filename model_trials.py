import keras
from keras.layers import Input, Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization, \
    Activation, Add, concatenate
from keras.models import Model
from tensorflow.python.ops.init_ops_v2 import glorot_uniform

from inception import inception_module_wout_bn_wd, inception_module_with_bn_wout_wd,\
    inception_module_with_bn_wd, inception_module_wout_bn_with_wd
from resnet import bottleneck_residual_block_with_bn_wout_wd, \
    bottleneck_residual_block_wout_bn_wd, preactivated_residual_block_wout_bn_wd ,\
    preactivated_residual_block_with_bn_wout_wd

#----------------------------------------------------------
# model 1
# 6 stages , no reg, no batch norm, no skip
def model1(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = Activation('relu', name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = Activation('relu', name='st1_relu2')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv3')(X)
    X = Activation('relu', name='st1_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = Activation('relu', name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = Activation('relu', name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st3_conv1')(X)
    X = Activation('relu', name='st3_relu1')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool')(X)

    # forth stage
    X = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st4_conv1')(X)
    X = Activation('relu', name='st4_relu1')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool')(X)

    # fifth stage
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv1')(X)
    X = Activation('relu', name='st5_relu1')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool')(X)

    # sixth stage
    X = Flatten(name='Flatten')(X)
    X = Dense(512, activation='relu', name='dense-1')(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model 1 with batch norm
# 6 stages , no reg,no skip
def model1_bn(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu', name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu', name='st1_relu2')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv3')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm3')(X)
    X = Activation('relu', name='st1_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu', name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu', name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st3_conv1')(X)
    X = BatchNormalization(axis=3, name='st3_batch_norm1')(X)
    X = Activation('relu', name='st3_relu1')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool')(X)

    # forth stage
    X = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st4_conv1')(X)
    X = BatchNormalization(axis=3, name='st4_batch_norm1')(X)
    X = Activation('relu', name='st4_relu1')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool')(X)

    # fifth stage
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv1')(X)
    X = BatchNormalization(axis=3, name='st5_batch_norm1')(X)
    X = Activation('relu', name='st5_relu1')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool')(X)

    # sixth stage
    X = Flatten(name='Flatten')(X)
    X = Dense(512, activation='relu', name='dense-1')(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model 1 with batch norm with gender info
# 6 stages , no reg,no skip
def model1_bn_gn(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu', name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu', name='st1_relu2')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv3')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm3')(X)
    X = Activation('relu', name='st1_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu', name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu', name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st3_conv1')(X)
    X = BatchNormalization(axis=3, name='st3_batch_norm1')(X)
    X = Activation('relu', name='st3_relu1')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool')(X)

    # forth stage
    X = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st4_conv1')(X)
    X = BatchNormalization(axis=3, name='st4_batch_norm1')(X)
    X = Activation('relu', name='st4_relu1')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool')(X)

    # fifth stage
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv1')(X)
    X = BatchNormalization(axis=3, name='st5_batch_norm1')(X)
    X = Activation('relu', name='st5_relu1')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool')(X)

    # sixth stage
    X = Flatten(name='Flatten')(X)

    Y_input = Input(1)
    Y = Dense(100, activation='relu', name='dense-gender', kernel_initializer=glorot_uniform(seed=0))(Y_input)

    X = concatenate(inputs=[X, Y])

    X = Dense(512, activation='relu', name='dense-1')(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=[X_input, Y_input], outputs=X, name='trial')

    return model

#----------------------------------------------------------

# model2
# 6 stages , no reg, no batch norm, no skip, yes inception
def model2(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = Activation('relu', name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = Activation('relu', name='st1_relu2')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv3')(X)
    X = Activation('relu', name='st1_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = Activation('relu', name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = Activation('relu', name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_wout_bn_wd(X,
                                    filters_1x1=32,
                                    filters_3x3_reduce=16,
                                    filters_3x3=32,
                                    filters_5x5_reduce=16,
                                    filters_5x5=32,
                                    filters_pool_proj=32,
                                    name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool')(X)

    # forth stage
    X = inception_module_wout_bn_wd(X,
                                    filters_1x1=32,
                                    filters_3x3_reduce=32,
                                    filters_3x3=64,
                                    filters_5x5_reduce=16,
                                    filters_5x5=32,
                                    filters_pool_proj=32,
                                    name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool')(X)

    # fifth stage
    X = inception_module_wout_bn_wd(X,
                                    filters_1x1=32,
                                    filters_3x3_reduce=64,
                                    filters_3x3=128,
                                    filters_5x5_reduce=16,
                                    filters_5x5=32,
                                    filters_pool_proj=32,
                                    name='st5_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool')(X)

    # sixth stage
    X = Flatten()(X)
    X = Dense(512, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model2 with batch norm
# 6 stages , no reg, no skip, yes inception
def model2_bn(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu', name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu', name='st1_relu2')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv3')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm3')(X)
    X = Activation('relu', name='st1_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu', name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu', name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_with_bn_wout_wd(X,
                                         filters_1x1=32,
                                         filters_3x3_reduce=16,
                                         filters_3x3=32,
                                         filters_5x5_reduce=16,
                                         filters_5x5=32,
                                         filters_pool_proj=32,
                                         name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool')(X)

    # forth stage
    X = inception_module_with_bn_wout_wd(X,
                                         filters_1x1=32,
                                         filters_3x3_reduce=32,
                                         filters_3x3=64,
                                         filters_5x5_reduce=16,
                                         filters_5x5=32,
                                         filters_pool_proj=32,
                                         name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool')(X)

    # fifth stage
    X = inception_module_with_bn_wout_wd(X,
                                         filters_1x1=32,
                                         filters_3x3_reduce=64,
                                         filters_3x3=128,
                                         filters_5x5_reduce=16,
                                         filters_5x5=32,
                                         filters_pool_proj=32,
                                         name='st5_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool')(X)

    # sixth stage
    X = Flatten()(X)
    X = Dense(512, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model2 with batch norm with gender info
# 6 stages , no reg, no skip, yes inception
def model2_bn_gn(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu', name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu', name='st1_relu2')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv3')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm3')(X)
    X = Activation('relu', name='st1_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu', name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu', name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_with_bn_wout_wd(X,
                                         filters_1x1=32,
                                         filters_3x3_reduce=16,
                                         filters_3x3=32,
                                         filters_5x5_reduce=16,
                                         filters_5x5=32,
                                         filters_pool_proj=32,
                                         name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool')(X)

    # forth stage
    X = inception_module_with_bn_wout_wd(X,
                                         filters_1x1=32,
                                         filters_3x3_reduce=32,
                                         filters_3x3=64,
                                         filters_5x5_reduce=16,
                                         filters_5x5=32,
                                         filters_pool_proj=32,
                                         name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool')(X)

    # fifth stage
    X = inception_module_with_bn_wout_wd(X,
                                         filters_1x1=32,
                                         filters_3x3_reduce=64,
                                         filters_3x3=128,
                                         filters_5x5_reduce=16,
                                         filters_5x5=32,
                                         filters_pool_proj=32,
                                         name='st5_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool')(X)

    # sixth stage
    X = Flatten(name='Flatten')(X)

    Y_input = Input(1)
    Y = Dense(100, activation='relu', name='dense-gender', kernel_initializer=glorot_uniform(seed=0))(Y_input)

    X = concatenate(inputs=[X, Y])

    X = Dense(512, activation='relu', name='dense-1')(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=[X_input, Y_input], outputs=X, name='trial')

    return model


#----------------------------------------------------------
# model3
# 6 stages , no reg, no batch norm, yes skip
def model3(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu', name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu', name='st1_relu2')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv3')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm3')(X)
    X = Activation('relu', name='st1_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu', name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu', name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = bottleneck_residual_block_wout_bn_wd(X, 3, [32, 32, 32],
                                             stage=3, block='a', reduce=True, s=2)

    # forth stage
    X = bottleneck_residual_block_wout_bn_wd(X, 3, [64, 64, 64],
                                             stage=4, block='a', reduce=True, s=2)

    # fifth stage
    X = bottleneck_residual_block_wout_bn_wd(X, 3, [128, 128, 128],
                                             stage=5, block='a', reduce=True, s=2)

    # sixth stage
    X = Flatten()(X)
    X = Dense(512, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model3 with batch norm
# 6 stages , no reg, yes skip
def model3_bn(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu', name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu', name='st1_relu2')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv3')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm3')(X)
    X = Activation('relu', name='st1_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu', name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu', name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = bottleneck_residual_block_with_bn_wout_wd(X, 3, [32, 32, 32],
                                                  stage=3, block='a', reduce=True, s=2)

    # forth stage
    X = bottleneck_residual_block_with_bn_wout_wd(X, 3, [64, 64, 64],
                                                  stage=4, block='a', reduce=True, s=2)

    # fifth stage
    X = bottleneck_residual_block_with_bn_wout_wd(X, 3, [128, 128, 128],
                                                  stage=5, block='a', reduce=True, s=2)

    # sixth stage
    X = Flatten()(X)
    X = Dense(512, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model3 with batch norm with gender
# 6 stages , no reg, yes skip
def model3_bn_gn(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu', name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu', name='st1_relu2')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv3')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm3')(X)
    X = Activation('relu', name='st1_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu', name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu', name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = bottleneck_residual_block_with_bn_wout_wd(X, 3, [32, 32, 32],
                                                  stage=3, block='a', reduce=True, s=2)

    # forth stage
    X = bottleneck_residual_block_with_bn_wout_wd(X, 3, [64, 64, 64],
                                                  stage=4, block='a', reduce=True, s=2)

    # fifth stage
    X = bottleneck_residual_block_with_bn_wout_wd(X, 3, [128, 128, 128],
                                                  stage=5, block='a', reduce=True, s=2)

    # sixth stage
    X = Flatten(name='Flatten')(X)

    Y_input = Input(1)
    Y = Dense(100, activation='relu', name='dense-gender', kernel_initializer=glorot_uniform(seed=0))(Y_input)

    X = concatenate(inputs=[X, Y])

    X = Dense(512, activation='relu', name='dense-1')(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=[X_input, Y_input], outputs=X, name='trial')

    return model


#----------------------------------------------------------
# model4
# 6 stages , no reg, no batch norm, yes skip- pre activated
def model4(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = Activation('relu', name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = Activation('relu', name='st1_relu2')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv3')(X)
    X = Activation('relu', name='st1_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = Activation('relu', name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)

    # Do not use activation after second stage, because it is going to be pre applied in preactivated residual block
    # third stage
    X = preactivated_residual_block_wout_bn_wd(X, 3, [32, 32, 32],
                                               stage=3, block='a', reduce=True, s=2)

    # forth stage
    X = preactivated_residual_block_wout_bn_wd(X, 3, [64, 64, 64],
                                               stage=4, block='a', reduce=True, s=2)

    # fifth stage
    X = preactivated_residual_block_wout_bn_wd(X, 3, [128, 128, 128],
                                               stage=5, block='a', reduce=True, s=2)
    # Use max pool
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool')(X)

    # sixth stage
    X = Flatten()(X)
    X = Dense(512, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model4 with batch norm
# 6 stages , no reg, yes skip- pre activated
def model4_bn(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu', name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu', name='st1_relu2')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv3')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm3')(X)
    X = Activation('relu', name='st1_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu', name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)

    # Do not use activation after second stage, because it is going to be pre applied in preactivated residual block
    # third stage
    X = preactivated_residual_block_with_bn_wout_wd (X, 3, [32, 32, 32],
                                               stage=3, block='a', reduce=True, s=2)

    # forth stage
    X = preactivated_residual_block_with_bn_wout_wd(X, 3, [64, 64, 64],
                                               stage=4, block='a', reduce=True, s=2)

    # fifth stage
    X = preactivated_residual_block_with_bn_wout_wd(X, 3, [128, 128, 128],
                                               stage=5, block='a', reduce=True, s=2)
    # Use max pool
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool')(X)

    # sixth stage
    X = Flatten()(X)
    X = Dense(512, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model

# model4 with batch norm with gender info
# 6 stages , no reg, yes skip- pre activated
def model4_bn_gn(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu', name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu', name='st1_relu2')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv3')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm3')(X)
    X = Activation('relu', name='st1_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu', name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)

    # Do not use activation after second stage, because it is going to be pre applied in preactivated residual block
    # third stage
    X = preactivated_residual_block_with_bn_wout_wd (X, 3, [32, 32, 32],
                                               stage=3, block='a', reduce=True, s=2)

    # forth stage
    X = preactivated_residual_block_with_bn_wout_wd(X, 3, [64, 64, 64],
                                               stage=4, block='a', reduce=True, s=2)

    # fifth stage
    X = preactivated_residual_block_with_bn_wout_wd(X, 3, [128, 128, 128],
                                               stage=5, block='a', reduce=True, s=2)
    # Use max pool
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool')(X)

    # sixth stage
    X = Flatten(name='Flatten')(X)

    Y_input = Input(1)
    Y = Dense(100, activation='relu', name='dense-gender', kernel_initializer=glorot_uniform(seed=0))(Y_input)

    X = concatenate(inputs=[X, Y])

    X = Dense(512, activation='relu', name='dense-1')(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=[X_input, Y_input], outputs=X, name='trial')

    return model


#----------------------------------------------------------
# model5
# 6 stages , no reg, no batch norm, yes skip, yes inception
def model5(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = Activation('relu', name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = Activation('relu', name='st1_relu2')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv3')(X)
    X = Activation('relu', name='st1_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = Activation('relu', name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = Activation('relu', name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    X_shortcut = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        name='st3_skip_conn_conv1')(X_shortcut)
    X = inception_module_wout_bn_wd(X,
                                    filters_1x1=32,
                                    filters_3x3_reduce=16,
                                    filters_3x3=32,
                                    filters_5x5_reduce=16,
                                    filters_5x5=32,
                                    filters_pool_proj=32,
                                    name='st3_inception1_')
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool')(X)

    # forth stage
    X_shortcut = X
    X_shortcut = Conv2D(filters=160, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        name='st4_skip_conn_conv1')(X_shortcut)
    X = inception_module_wout_bn_wd(X,
                                    filters_1x1=32,
                                    filters_3x3_reduce=32,
                                    filters_3x3=64,
                                    filters_5x5_reduce=16,
                                    filters_5x5=32,
                                    filters_pool_proj=32,
                                    name='st4_inception1_')
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool')(X)

    # fifth stage
    X_shortcut = X
    X_shortcut = Conv2D(filters=224, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        name='st5_skip_conn_conv1')(X_shortcut)
    X = inception_module_wout_bn_wd(X,
                                    filters_1x1=32,
                                    filters_3x3_reduce=64,
                                    filters_3x3=128,
                                    filters_5x5_reduce=16,
                                    filters_5x5=32,
                                    filters_pool_proj=32,
                                    name='st5_inception1_')
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool')(X)

    # sixth stage
    X = Flatten()(X)
    X = Dense(512, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model5 with batch norm
# 6 stages , no reg yes skip, yes inception
def model5_bn(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu', name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu', name='st1_relu2')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv3')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm3')(X)
    X = Activation('relu', name='st1_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu', name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu', name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)


    # third stage
    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    X_shortcut = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        name='st3_skip_conn_conv1')(X_shortcut)
    X = inception_module_with_bn_wout_wd(X,
                                    filters_1x1=32,
                                    filters_3x3_reduce=16,
                                    filters_3x3=32,
                                    filters_5x5_reduce=16,
                                    filters_5x5=32,
                                    filters_pool_proj=32,
                                    name='st3_inception1_')
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool')(X)

    # forth stage
    X_shortcut = X
    X_shortcut = Conv2D(filters=160, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        name='st4_skip_conn_conv1')(X_shortcut)
    X = inception_module_with_bn_wout_wd(X,
                                    filters_1x1=32,
                                    filters_3x3_reduce=32,
                                    filters_3x3=64,
                                    filters_5x5_reduce=16,
                                    filters_5x5=32,
                                    filters_pool_proj=32,
                                    name='st4_inception1_')
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool')(X)

    # fifth stage
    X_shortcut = X
    X_shortcut = Conv2D(filters=224, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        name='st5_skip_conn_conv1')(X_shortcut)
    X = inception_module_with_bn_wout_wd(X,
                                    filters_1x1=32,
                                    filters_3x3_reduce=64,
                                    filters_3x3=128,
                                    filters_5x5_reduce=16,
                                    filters_5x5=32,
                                    filters_pool_proj=32,
                                    name='st5_inception1_')
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool')(X)

    # sixth stage
    X = Flatten()(X)
    X = Dense(512, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model5 with batch norm with gender info
# 6 stages , no reg yes skip, yes inception
def model5_bn_gn(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu', name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu', name='st1_relu2')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv3')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm3')(X)
    X = Activation('relu', name='st1_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu', name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu', name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)


    # third stage
    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X
    X_shortcut = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        name='st3_skip_conn_conv1')(X_shortcut)
    X = inception_module_with_bn_wout_wd(X,
                                    filters_1x1=32,
                                    filters_3x3_reduce=16,
                                    filters_3x3=32,
                                    filters_5x5_reduce=16,
                                    filters_5x5=32,
                                    filters_pool_proj=32,
                                    name='st3_inception1_')
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool')(X)

    # forth stage
    X_shortcut = X
    X_shortcut = Conv2D(filters=160, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        name='st4_skip_conn_conv1')(X_shortcut)
    X = inception_module_with_bn_wout_wd(X,
                                    filters_1x1=32,
                                    filters_3x3_reduce=32,
                                    filters_3x3=64,
                                    filters_5x5_reduce=16,
                                    filters_5x5=32,
                                    filters_pool_proj=32,
                                    name='st4_inception1_')
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool')(X)

    # fifth stage
    X_shortcut = X
    X_shortcut = Conv2D(filters=224, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        name='st5_skip_conn_conv1')(X_shortcut)
    X = inception_module_with_bn_wout_wd(X,
                                    filters_1x1=32,
                                    filters_3x3_reduce=64,
                                    filters_3x3=128,
                                    filters_5x5_reduce=16,
                                    filters_5x5=32,
                                    filters_pool_proj=32,
                                    name='st5_inception1_')
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool')(X)

    # sixth stage
    X = Flatten(name='Flatten')(X)

    Y_input = Input(1)
    Y = Dense(100, activation='relu', name='dense-gender', kernel_initializer=glorot_uniform(seed=0))(Y_input)

    X = concatenate(inputs=[X, Y])

    X = Dense(512, activation='relu', name='dense-1')(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=[X_input, Y_input], outputs=X, name='trial')

    return model


#----------------------------------------------------------
# model transfer_learning
# use model6 as translearning base model
# model 6 as base_model + 2 extra residual layers +dense
def model_transfer_learning():
    path_real = './my_models/'
    base_model = keras.models.load_model(path_real + 'model6.h5')

    # Extract the output of the last convolutional layer
    feature_extractor = Model(inputs=base_model.input,
                              outputs=base_model.get_layer(name='act6a_branch_last_relu').output)

    # Fine-tune from this layer onwards
    fine_tune_at = -12

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in feature_extractor.layers[:fine_tune_at]:
        layer.trainable = False

    X = feature_extractor.output
    # Add new layers on top of the feature extractor

    # seventh stage
    X = inception_module_wout_bn_wd(X,
                                    filters_1x1=64,
                                    filters_3x3_reduce=32,
                                    filters_3x3=96,
                                    filters_5x5_reduce=16,
                                    filters_5x5=64,
                                    filters_pool_proj=64,
                                    name='st7_inception1_')

    X = inception_module_wout_bn_wd(X,
                                    filters_1x1=64,
                                    filters_3x3_reduce=64,
                                    filters_3x3=128,
                                    filters_5x5_reduce=32,
                                    filters_5x5=64,
                                    filters_pool_proj=64,
                                    name='st7_inception2_')

    X = Flatten(name="flatten")(X)
    X = Dropout(0.5)(X)
    X = Dense(1024, activation='relu', name='dense1')(X)
    X = Dropout(0.5)(X)
    X = Dense(1, name='dense_final')(X)

    # Create a new model with the feature extractor and the new layers
    model = Model(inputs=feature_extractor.input, outputs=X)

    return model


#----------------------------------------------------------
#model = model1_bn(input_shape=(224, 224, 1))
#num_params = model.count_params()
#print(f'Number of parameters: {num_params:,}\n')
#model.summary()
#
