from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, \
    MaxPooling2D
from keras.models import Model
from tensorflow.python.ops.init_ops_v2 import glorot_uniform
from keras.regularizers import L2


# Residual block with  batch normaliation and weightdecay
def bottleneck_residual_block_with_bn_wd(X, f, filters, stage, block, reduce=False, s=2):
    """
    Arguments:
    X -- input tensor of shape (m, height, width, channels)
    f -- integer, specifying the shape of the middle CONV's kernel window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    reduce -- boolean, True = identifies the reduction layer at the beginning of each learning stage
    s -- integer, strides

    Returns:
    X -- output of the identity block, tensor of shape (H, W, C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    if reduce:
        # if we are to reduce the spatial size, apply a 1x1 CONV layer to the shortcut path
        # to do that, we need both CONV layers to have similar strides
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=L2(0.0001),
                   name=conv_name_base + '2a')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                            kernel_regularizer=L2(0.0001), name=conv_name_base + '1')(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
    else:
        # First component of main path
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=L2(0.0001),
                   name=conv_name_base + '2a')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_regularizer=L2(0.0001),
               name=conv_name_base + '2b')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=L2(0.0001),
               name=conv_name_base + '2c')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X



# Residual block with batch normalization and without weightdecay
def bottleneck_residual_block_with_bn_wout_wd(X, f, filters, stage, block, reduce=False, s=2):
    """
    Arguments:
    X -- input tensor of shape (m, height, width, channels)
    f -- integer, specifying the shape of the middle CONV's kernel window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    reduce -- boolean, True = identifies the reduction layer at the beginning of each learning stage
    s -- integer, strides

    Returns:
    X -- output of the identity block, tensor of shape (H, W, C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    activation_name ='act' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    if reduce:
        # if we are to reduce the spatial size, apply a 1x1 CONV layer to the shortcut path
        # to do that, we need both CONV layers to have similar strides
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid',
                   name=conv_name_base + '2a')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu',name = activation_name + '2a')(X)

        X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                            kernel_regularizer=L2(0.0001), name=conv_name_base + '1')(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
    else:
        # First component of main path
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   name=conv_name_base + '2a')(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu',name = activation_name + '2a')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu',name = activation_name + '2b')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c')(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add(name = 'add' + str(stage) + block )([X, X_shortcut])
    X = Activation('relu', name=activation_name + '_last_relu')(X)

    return X


# Residual block without batch normalization and with weightdecay
def bottleneck_residual_block_wout_bn_with_wd(X, f, filters, stage, block, reduce=False, s=2):
    """
    Arguments:
    X -- input tensor of shape (m, height, width, channels)
    f -- integer, specifying the shape of the middle CONV's kernel window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    reduce -- boolean, True = identifies the reduction layer at the beginning of each learning stage
    s -- integer, strides

    Returns:
    X -- output of the identity block, tensor of shape (H, W, C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    if reduce:
        # if we are to reduce the spatial size, apply a 1x1 CONV layer to the shortcut path
        # to do that, we need both CONV layers to have similar strides
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=L2(0.0001),
                   name=conv_name_base + '2a')(X)
        X = Activation('relu')(X)

        X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                            kernel_regularizer=L2(0.0001), name=conv_name_base + '1')(X_shortcut)
    else:
        # First component of main path
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=L2(0.0001),
                   name=conv_name_base + '2a')(X)
        X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', kernel_regularizer=L2(0.0001),
               name=conv_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=L2(0.0001),
               name=conv_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


# Residual block without batch normalization and weightdecay
def bottleneck_residual_block_wout_bn_wd(X, f, filters, stage, block, reduce=False, s=2):
    """
    Arguments:
    X -- input tensor of shape (m, height, width, channels)
    f -- integer, specifying the shape of the middle CONV's kernel window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    reduce -- boolean, True = identifies the reduction layer at the beginning of each learning stage
    s -- integer, strides

    Returns:
    X -- output of the identity block, tensor of shape (H, W, C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    if reduce:
        # if we are to reduce the spatial size, apply a 1x1 CONV layer to the shortcut path
        # to do that, we need both CONV layers to have similar strides
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a')(X)
        X = Activation('relu')(X)

        X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1')(
            X_shortcut)
    else:
        # First component of main path
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a')(X)
        X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


# Preactivated residual block with batch normalization and without weightdecay
def preactivated_residual_block_with_bn_wout_wd(X, f, filters, stage, block, reduce=False, s=2):
    """
    Arguments:
    X -- input tensor of shape (m, height, width, channels)
    f -- integer, specifying the shape of the middle CONV's kernel window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    reduce -- boolean, True = identifies the reduction layer at the beginning of each learning stage
    s -- integer, strides

    Returns:
    X -- output of the identity block, tensor of shape (H, W, C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    if reduce:
        # if we are to reduce the spatial size, apply a 1x1 CONV layer to the shortcut path
        # to do that, we need both CONV layers to have similar strides
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid',
                   name=conv_name_base + '2a')(X)

        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
        X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                            kernel_regularizer=L2(0.0001), name=conv_name_base + '1')(X_shortcut)

    else:
        # First component of main path
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   name=conv_name_base + '2a')(X)


    # Second component of main path

    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b')(X)


    # Third component of main path
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c')(X)


    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])


    return X


# Preactivated residual block without batch normalization and weightdecay
def preactivated_residual_block_wout_bn_wd(X, f, filters, stage, block, reduce=False, s=2):
    """
    Arguments:
    X -- input tensor of shape (m, height, width, channels)
    f -- integer, specifying the shape of the middle CONV's kernel window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    reduce -- boolean, True = identifies the reduction layer at the beginning of each learning stage
    s -- integer, strides

    Returns:
    X -- output of the identity block, tensor of shape (H, W, C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    if reduce:
        # if we are to reduce the spatial size, apply a 1x1 CONV layer to the shortcut path
        # to do that, we need both CONV layers to have similar strides
        X = Activation('relu')(X)
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid',
                   name=conv_name_base + '2a')(X)

        X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                            kernel_regularizer=L2(0.0001), name=conv_name_base + '1')(X_shortcut)

    else:
        # First component of main path
        X = Activation('relu')(X)
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   name=conv_name_base + '2a')(X)


    # Second component of main path

    X = Activation('relu')(X)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
               name=conv_name_base + '2b')(X)


    # Third component of main path
    X = Activation('relu')(X)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
               name=conv_name_base + '2c')(X)


    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])


    return X


