import keras.layers
from keras.layers import Input, Conv2D, MaxPool2D, AveragePooling2D, Dense, Dropout, Flatten, BatchNormalization, \
    Activation, Add, concatenate
from keras.models import Model
from keras.regularizers import L2
from tensorflow.python.ops.init_ops_v2 import glorot_uniform

from inception import inception_module_without_bn_without_weightdecay,\
    inception_module_with_bn, inception_module_without_bn, inception_module_without_weigth_decay

from resnet import bottleneck_residual_block_without_bn_without_weightdecay,\
    bottleneck_residual_block_with_bn, bottleneck_residual_block_without_bn, \
    bottleneck_residual_block_with_bn_without_weightdecay,preactivated_residual_block_with_bn_without_weightdecay


# model 1
#6 stages , no reg, no batch norm, no skip
def model1(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_without_bn_without_weightdecay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    X = inception_module_without_bn_without_weightdecay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)

    # fifth stage
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv1')(X)
    X = Activation('relu',name='st5_relu1')(X)
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv2')(X)
    X = Activation('relu',name='st5_relu2')(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv3')(X)
    X = Activation('relu',name='st5_relu3')(X)

    X = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid', name='average_pool')(X)

    # sixth stage
    X = Flatten(name='Flatten')(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model2
#8 stages , no reg, no batch norm, no skip
def model2(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_without_bn_without_weightdecay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    X = inception_module_without_bn_without_weightdecay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)

    # fifth stage
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv1')(X)
    X = Activation('relu',name='st5_relu1')(X)
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv2')(X)
    X = Activation('relu',name='st5_relu2')(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv3')(X)
    X = Activation('relu',name='st5_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool1')(X)


    # sixth stage
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st6_conv1')(X)
    X = Activation('relu',name='st6_relu1')(X)
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st6_conv2')(X)
    X = Activation('relu',name='st6_relu2')(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st6_conv3')(X)
    X = Activation('relu',name='st6_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st6_max_pool1')(X)


    # seventh stage
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st7_conv1')(X)
    X = Activation('relu',name='st7_relu1')(X)
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st7_conv2')(X)
    X = Activation('relu',name='st7_relu2')(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st7_conv3')(X)
    X = Activation('relu',name='st7_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st7_max_pool1')(X)

    # eighth stage
    X = Flatten()(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model

# model3
#7 stages , no reg, no batch norm, no skip
def model3(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_without_bn_without_weightdecay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    X = inception_module_without_bn_without_weightdecay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)

    # fifth stage
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv1')(X)
    X = Activation('relu',name='st5_relu1')(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv2')(X)
    X = Activation('relu',name='st5_relu2')(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv3')(X)
    X = Activation('relu',name='st5_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool1')(X)


    # sixth stage
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st6_conv1')(X)
    X = Activation('relu',name='st6_relu1')(X)
    X = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st6_conv2')(X)
    X = Activation('relu',name='st6_relu2')(X)
    X = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st6_conv3')(X)
    X = Activation('relu',name='st6_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st6_max_pool1')(X)

    # seventh stage
    X = Flatten()(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model 4
# 7 stages, yes reg, no batch norm, no skip
def model4(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1',kernel_regularizer=L2(0.0001), kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_regularizer=L2(0.0001), name='st1_conv2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_regularizer=L2(0.0001),
               name='st2_conv1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_regularizer=L2(0.0001),
               name='st2_conv2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_without_bn_without_weightdecay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    X = inception_module_without_bn_without_weightdecay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)

    # fifth stage
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_regularizer=L2(0.0001), name='st5_conv1')(X)
    X = Activation('relu',name='st5_relu1')(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_regularizer=L2(0.0001), name='st5_conv2')(X)
    X = Activation('relu',name='st5_relu2')(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_regularizer=L2(0.0001), name='st5_conv3')(X)
    X = Activation('relu',name='st5_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool1')(X)


    # sixth stage
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_regularizer=L2(0.0001), name='st6_conv1')(X)
    X = Activation('relu',name='st6_relu1')(X)
    X = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_regularizer=L2(0.0001), name='st6_conv2')(X)
    X = Activation('relu',name='st6_relu2')(X)
    X = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same',kernel_regularizer=L2(0.0001), name='st6_conv3')(X)
    X = Activation('relu',name='st6_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st6_max_pool1')(X)

    # seventh stage
    X = Flatten()(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=L2(0.0001))(X)
    X = Dropout(0.5, name='dropout-1')(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dropout(0.5, name='dropout-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model

# model 5
# 7 stages, yes batch norm, no reg, no skip
def model5(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)

    # fifth stage
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv1')(X)
    X = BatchNormalization(axis=3, name='st5_batch_norm1')(X)
    X = Activation('relu',name='st5_relu1')(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv2')(X)
    X = BatchNormalization(axis=3, name='st5_batch_norm2')(X)
    X = Activation('relu',name='st5_relu2')(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv3')(X)
    X = BatchNormalization(axis=3, name='st5_batch_norm3')(X)
    X = Activation('relu',name='st5_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool1')(X)


    # sixth stage
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st6_conv1')(X)
    X = BatchNormalization(axis=3, name='st6_batch_norm1')(X)
    X = Activation('relu',name='st6_relu1')(X)
    X = Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st6_conv2')(X)
    X = BatchNormalization(axis=3, name='st6_batch_norm2')(X)
    X = Activation('relu',name='st6_relu2')(X)
    X = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st6_conv3')(X)
    X = BatchNormalization(axis=3, name='st6_batch_norm3')(X)
    X = Activation('relu',name='st6_relu3')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st6_max_pool1')(X)

    # seventh stage
    X = Flatten()(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model 6
# 7 stages, yes batch norm, yes skip, no reg
def model6(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)



    # fifth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 256], stage=5, block='a', reduce=True,
                                                                 s=2)

    # sixth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 512], stage=6, block='a',reduce=True,
                                                              s=2)

    # seventh stage
    X = Flatten()(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model 7
# 9 stages, yes batch norm, yes skip, no reg
def model7(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)



    # fifth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 256], stage=5, block='a', reduce=True,
                                                                 s=2)

    # sixth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 512], stage=6, block='a',reduce=True,
                                                              s=2)

    # seventh stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [512, 512, 512], stage=7, block='a', reduce=True,
                                                              s=2)

    # eight stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [512, 512, 1024], stage=8, block='a', reduce=True,
                                                              s=2)
    # ninth stage
    X = Flatten()(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model 8
# 9 stages, yes batch norm, yes skip, yes inception skip, no reg
def model8(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    # skip connection for inception module
    X_shortcut = X
    X_shortcut = Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='same',
                         name='st3_skip_conn_conv1')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name= 'st3_skip_conn_batch_norm1')(X_shortcut)


    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = Add(name='st3_skip1')([X, X_shortcut])
    X = Activation('relu',name = 'st3_relu')(X)

    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    # skip connection for inception module
    X_shortcut = X
    X_shortcut = Conv2D(filters=352, kernel_size=(1, 1), strides=(1, 1), padding='same',
                         name='st4_skip_conn_conv1')(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name= 'st4_skip_conn_batch_norm1')(X_shortcut)

    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')
    X = Add(name='st4_skip1')([X, X_shortcut])
    X = Activation('relu',name = 'st4_relu')(X)

    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)



    # fifth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 256], stage=5, block='a', reduce=True,
                                                                 s=2)

    # sixth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 512], stage=6, block='a',reduce=True,
                                                              s=2)

    # seventh stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [512, 512, 512], stage=7, block='a', reduce=True,
                                                              s=2)

    # eight stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [512, 512, 1024], stage=8, block='a', reduce=True,
                                                              s=2)
    # ninth stage
    X = Flatten(name='Flatten')(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model



# model 9
# 9 stages, yes batch norm, yes skip, no reg, extra skips for stage 5-6
def model9(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')

    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)



    # fifth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 512], stage=5, block='a', reduce=True,
                                                                 s=2)
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 512], stage=5, block='b', reduce=False)

    # sixth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [576, 576, 1024], stage=6, block='a',reduce=True,
                                                              s=2)
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [1024, 1024, 1024], stage=6, block='b', reduce=False)


    # sixt stage
    X = Flatten(name='Flatten')(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model 10
# 7 stages, yes batch norm, yes skip, yes dropout, no weight decay
def model10(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)



    # fifth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 256], stage=5, block='a', reduce=True,
                                                                 s=2)

    # sixth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 512], stage=6, block='a',reduce=True,
                                                              s=2)

    # seventh stage
    X = Flatten()(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(512, activation='relu', name='dense-2')(X)
    X = Dropout(0.5, name='dropout-1')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model 11
# 7 stages, yes batch norm, yes skip, yes dropout, yes weight decay
def model11(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu',name='st1_relu2')(X)


    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st3_conv1')(X)
    X = BatchNormalization(axis=3, name='st3_batch_norm1')(X)
    X = Activation('relu',name='st3_relu1')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st3_conv2')(X)
    X = BatchNormalization(axis=3, name='st3_batch_norm2')(X)
    X = Activation('relu',name='st3_relu2')(X)


    # forth stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st4_conv1')(X)
    X = BatchNormalization(axis=3, name='st4_batch_norm1')(X)
    X = Activation('relu',name='st4_relu1')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st4_conv2')(X)
    X = BatchNormalization(axis=3, name='st4_batch_norm2')(X)
    X = Activation('relu',name='st4_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool')(X)


    # fifthstage
    X = inception_module_with_bn(X,
                                                        filters_1x1=32,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=32,
                                                        filters_pool_proj=32,
                                                        name='st5_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool')(X)

    # sixth stage
    X = inception_module_with_bn(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=96,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=32,
                                                        filters_pool_proj=64,
                                                        name='st6_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st6_max_pool1')(X)



    # seventh stage
    X = bottleneck_residual_block_with_bn(X, 3, [256, 256, 256], stage=7, block='a', reduce=True,
                                                                 s=2)

    # eighth stage
    X = bottleneck_residual_block_with_bn(X, 3, [256, 256, 512], stage=8, block='a',reduce=True,
                                                              s=2)

    # ninth stage
    X = Flatten()(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model



# model 12
# 7 stages, yes batch norm, yes skip, no reg
def model12(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)



    # fifth stage
    X = preactivated_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 256], stage=5, block='a', reduce=True,
                                                                 s=2)

    # sixth stage
    X = preactivated_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 512], stage=6, block='a',reduce=True,
                                                              s=2)

    # seventh stage
    X = Flatten()(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model




# model 13
# 7 stages, yes batch norm, yes skip, no reg
def model13(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)



    # fifth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 256], stage=5, block='a', reduce=True,
                                                                 s=2)
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 256], stage=5, block='b',reduce=False)


    # sixth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 512], stage=6, block='a',reduce=True,
                                                              s=2)
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [512, 512, 512], stage=6, block='b',reduce=False)


    # seventh stage
    X = Flatten()(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model

def model_yes_reg_bn_no_skip(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1',kernel_regularizer=L2(0.0001), kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2',kernel_regularizer=L2(0.0001))(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1',kernel_regularizer=L2(0.0001))(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2',kernel_regularizer=L2(0.0001))(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_with_bn(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    X = inception_module_with_bn(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)

    # fifth stage
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv1',kernel_regularizer=L2(0.0001))(X)
    X = BatchNormalization(axis=3, name='st5_batch_norm1')(X)
    X = Activation('relu',name='st5_relu1')(X)
    X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv2',kernel_regularizer=L2(0.0001))(X)
    X = BatchNormalization(axis=3, name='st5_batch_norm2')(X)
    X = Activation('relu',name='st5_relu2')(X)
    X = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st5_conv3',kernel_regularizer=L2(0.0001))(X)
    X = BatchNormalization(axis=3, name='st5_batch_norm3')(X)
    X = Activation('relu',name='st5_relu3')(X)

    X = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid', name='average_pool')(X)

    X = Flatten(name='Flatten')(X)
    X = Dropout(0.5, name='dropout-1')(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dropout(0.5, name='dropout-2')(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dropout(0.2, name='dropout-3')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model



# model 14
# 7 stages, yes batch norm, yes skip, no reg, no inception
def model14(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [128, 128, 128], stage=3, block='a', reduce=True,
                                                                 s=2)

    # forth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [128, 128, 256], stage=4, block='a', reduce=True,
                                                                 s=2)



    # fifth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 256], stage=5, block='a', reduce=True,
                                                                 s=2)

    # sixth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 512], stage=6, block='a',reduce=True,
                                                              s=2)

    # seventh stage
    X = Flatten()(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model 15
# 7 stages, yes batch norm, no skip, no reg
def model15(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)

    # fifth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=128,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=256,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=128,
                                                        filters_pool_proj=64,
                                                        name='st5_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool1')(X)


    # sixth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=256,
                                                        filters_3x3_reduce=128,
                                                        filters_3x3=512,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=128,
                                                        filters_pool_proj=256,
                                                        name='st6_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st6_max_pool1')(X)


    # seventh stage
    X = Flatten()(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model



# model 16
# 7 stages, yes batch norm, no skip, no reg, model 15 with extra inception no extra max pool
def model16(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)

    # fifth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=128,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=256,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=128,
                                                        filters_pool_proj=64,
                                                        name='st5_inception1_')
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=128,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=256,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=128,
                                                        filters_pool_proj=64,
                                                        name='st5_inception2_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool1')(X)


    # sixth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=256,
                                                        filters_3x3_reduce=128,
                                                        filters_3x3=512,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=128,
                                                        filters_pool_proj=256,
                                                        name='st6_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st6_max_pool1')(X)


    # seventh stage
    X = Flatten()(X)
    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=X_input, outputs=X, name='trial')

    return model


# model 17
# 7 stages, yes batch norm, no skip, no reg , gender info
def model17(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)

    # fifth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=128,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=256,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=128,
                                                        filters_pool_proj=64,
                                                        name='st5_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st5_max_pool1')(X)


    # sixth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=256,
                                                        filters_3x3_reduce=128,
                                                        filters_3x3=512,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=128,
                                                        filters_pool_proj=256,
                                                        name='st6_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st6_max_pool1')(X)

    Y_input = Input(1)
    Y = Dense(100, activation='relu', name='dense-gender', kernel_initializer=glorot_uniform(seed=0))(Y_input)

    # seventh stage
    X = Flatten()(X)

    X = concatenate(inputs = [X,Y])

    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=[X_input,Y_input], outputs=X, name='trial')

    return model



# model 18
# 7 stages, yes batch norm, yes skip, no reg, gender info
def model18(input_shape):
    X_input = Input(input_shape, name='Input')

    # first stage
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st1_conv1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=3, name='st1_batch_norm1')(X)
    X = Activation('relu',name='st1_relu1')(X)
    X = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='st1_conv2')(X)
    X = BatchNormalization(axis=3, name='st1_batch_norm2')(X)
    X = Activation('relu',name='st1_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st1_max_pool1')(X)

    # second stage
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv1')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm1')(X)
    X = Activation('relu',name='st2_relu1')(X)
    X = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
               name='st2_conv2')(X)
    X = BatchNormalization(axis=3, name='st2_batch_norm2')(X)
    X = Activation('relu',name='st2_relu2')(X)
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st2_max_pool')(X)

    # third stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=64,
                                                        filters_3x3_reduce=32,
                                                        filters_3x3=64,
                                                        filters_5x5_reduce=16,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st3_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st3_max_pool1')(X)

    # forth stage
    X = inception_module_without_weigth_decay(X,
                                                        filters_1x1=96,
                                                        filters_3x3_reduce=64,
                                                        filters_3x3=128,
                                                        filters_5x5_reduce=32,
                                                        filters_5x5=64,
                                                        filters_pool_proj=64,
                                                        name='st4_inception1_')
    X = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='st4_max_pool1')(X)



    # fifth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 256], stage=5, block='a', reduce=True,
                                                                 s=2)

    # sixth stage
    X = bottleneck_residual_block_with_bn_without_weightdecay(X, 3, [256, 256, 512], stage=6, block='a',reduce=True,
                                                              s=2)

    # seventh stage
    X = Flatten()(X)

    Y_input = Input(1)
    Y = Dense(32, activation='relu', name='dense-gender', kernel_initializer=glorot_uniform(seed=0))(Y_input)

    X = concatenate(inputs=[X, Y])

    X = Dense(1000, activation='relu', name='dense-1',
              kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(256, activation='relu', name='dense-2')(X)
    X = Dense(1, name='output')(X)

    # Create the model
    model = Model(inputs=[X_input,Y_input], outputs=X, name='trial')

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

model = model17(input_shape=(224, 224, 3))
num_params = model.count_params()
print(f'Number of parameters: {num_params:,}\n')
model.summary()
