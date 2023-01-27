from keras.layers import Conv2D, MaxPool2D, \
    Dropout, Dense, Input, concatenate, AveragePooling2D, BatchNormalization, Activation
from keras.models import Model
from keras.regularizers import L2


def inception_module_wout_bn_with_wd(X,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu',kernel_regularizer=L2(0.0001),name = name + 'conv_1-1')(X)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu',kernel_regularizer=L2(0.0001),name = name + 'conv_3-3_reduce')(X)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu',kernel_regularizer=L2(0.0001),name = name + 'conv_3-3')(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu',kernel_regularizer=L2(0.0001),name = name + 'conv_5-5_reduce')(X)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu',kernel_regularizer=L2(0.0001),name = name + 'conv_5-5')(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same',name = name + '_max_pool')(X)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu',kernel_regularizer=L2(0.0001),name = name + '_max_pool_reduce')(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    return output

def inception_module_wout_bn_wd(X,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu',name = name + 'conv_1-1')(X)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu',name = name + 'conv_3-3_reduce')(X)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu',name = name + 'conv_3-3')(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu',name = name + 'conv_5-5_reduce')(X)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu',name = name + 'conv_5-5')(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same',name = name + '_max_pool')(X)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu',name = name + '_max_pool_reduce')(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    return output

def inception_module_with_bn_wd(X,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):

    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same',kernel_regularizer=L2(0.0001),name = name + 'conv_1-1')(X)
    conv_1x1 = BatchNormalization(axis=3,name=name + 'batch_norm_1-1')(conv_1x1)
    conv_1x1 = Activation('relu',name=name + 'relu_1-1')(conv_1x1)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same',kernel_regularizer=L2(0.0001),name = name + 'conv_3-3_reduce')(X)
    conv_3x3 = BatchNormalization(axis=3,name=name + 'batch_norm_3-3_reduce')(conv_3x3)
    conv_3x3 = Activation('relu',name=name + 'relu_3-3_reduce')(conv_3x3)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same',kernel_regularizer=L2(0.0001),name = name + 'conv_3-3')(conv_3x3)
    conv_3x3 = BatchNormalization(axis=3,name=name + 'batch_norm_3-3')(conv_3x3)
    conv_3x3 = Activation('relu',name=name + 'relu_3-3')(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same',kernel_regularizer=L2(0.0001),name = name + 'conv_5-5_reduce')(X)
    conv_5x5 = BatchNormalization(axis=3,name=name + 'batch_norm_5-5_reduce')(conv_5x5)
    conv_5x5 = Activation('relu',name=name + 'relu_5-5_reduce')(conv_5x5)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same',kernel_regularizer=L2(0.0001),name = name + 'conv_5-5')(conv_5x5)
    conv_5x5 = BatchNormalization(axis=3,name=name + 'batch_norm_5-5')(conv_5x5)
    conv_5x5 = Activation('relu',name=name + 'relu_5-5')(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same',name = name + '_max_pool')(X)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same',kernel_regularizer=L2(0.0001),name = name + '_max_pool_reduce')(pool_proj)
    pool_proj = BatchNormalization(axis=3,name=name + 'batch_norm_pool_reduce')(pool_proj)
    pool_proj = Activation('relu',name=name + 'relu_pool_reduce')(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name + 'concat')

    return output


def inception_module_with_bn_wout_wd(X,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):

    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same',name = name + 'conv_1-1')(X)
    conv_1x1 = BatchNormalization(axis=3,name=name + 'batch_norm_1-1')(conv_1x1)
    conv_1x1 = Activation('relu',name=name + 'relu_1-1')(conv_1x1)

    conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same',name = name + 'conv_3-3_reduce')(X)
    conv_3x3 = BatchNormalization(axis=3,name=name + 'batch_norm_3-3_reduce')(conv_3x3)
    conv_3x3 = Activation('relu',name=name + 'relu_3-3_reduce')(conv_3x3)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same',name = name + 'conv_3-3')(conv_3x3)
    conv_3x3 = BatchNormalization(axis=3,name=name + 'batch_norm_3-3')(conv_3x3)
    conv_3x3 = Activation('relu',name=name + 'relu_3-3')(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same',name = name + 'conv_5-5_reduce')(X)
    conv_5x5 = BatchNormalization(axis=3,name=name + 'batch_norm_5-5_reduce')(conv_5x5)
    conv_5x5 = Activation('relu',name=name + 'relu_5-5_reduce')(conv_5x5)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same',name = name + 'conv_5-5')(conv_5x5)
    conv_5x5 = BatchNormalization(axis=3,name=name + 'batch_norm_5-5')(conv_5x5)
    conv_5x5 = Activation('relu',name=name + 'relu_5-5')(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same',name = name + '_max_pool')(X)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same',name = name + '_max_pool_reduce')(pool_proj)
    pool_proj = BatchNormalization(axis=3,name=name + 'batch_norm_pool_reduce')(pool_proj)
    pool_proj = Activation('relu',name=name + 'relu_pool_reduce')(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name + 'concat')

    return output

