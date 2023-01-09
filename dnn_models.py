# Create Models

# DNN Model
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
    X_input = tf.keras.Input(input_shape)

    # FLATTEN THE TENSOR
    X = tf.keras.layers.Flatten()(X_input)

    X = tf.keras.layers.Dropout(0.5)(X)

    X = tf.keras.layers.Dense(128, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.5)(X)

    X = tf.keras.layers.Dense(64, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.5)(X)

    X = tf.keras.layers.Dense(32, activation='relu')(X)
    X = tf.keras.layers.Dropout(0.5)(X)

    X_output = tf.keras.layers.Dense(1)(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = tf.keras.Model(inputs=X_input, outputs=X_output, name='DNN_Model1')

    return model


# In[14]:


# NETWORK MODEL

dnn_model = DNN_Model((Input_Size, Input_Size, 3))

# In[15]:


num_params = dnn_model.count_params()
print(f'Number of parameters: {num_params:,}')

print_memory_info()

dnn_model.summary()

# In[17]:


# Create a callback that will interrupt training when the validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# Compile the model
dnn_model.compile(optimizer=adam_optimizer, loss=tf.keras.losses.MeanAbsoluteError(),
                  metrics=tf.keras.metrics.mean_absolute_error)

# Train the model
hist = dnn_model.fit(train_dataset, epochs=200,
                     validation_data=validation_dataset,
                     callbacks=[early_stopping],
                     use_multiprocessing=True,
                     workers=os.cpu_count()
                     )

# Save the model in HDF5 format
tf.keras.models.save_model(dnn_model, './denememodel/dnn_model.h5')
np.save('./denememodel/dnn_history.npy', hist.history)

# Restore the model from the HDF5 file
# model = tf.keras.models.load_model('./denememodel/model.h5')
# Get back the history
# history1=np.load('./denememodel/history1.npy',allow_pickle='TRUE').item()


print("\n")
print("-----------------------------------------------")

print_memory_info()

# In[18]:


# Evaluate the model
test_mae = dnn_model.evaluate(test_dataset, workers=-1)
print('Test loss:', test_mae)

print("\n")
print("-----------------------------------------------")

hist_graphs(hist)


# ### LeNet Model

# In[ ]:


def LeNet_Model(input_shape):
    # Define input tensor with specified shape
    X_input = tf.keras.Input(input_shape)

    # Add convolutional layer with 6 filters, kernel size of (5, 5), stride of (1, 1), and hyperbolic tangent activation
    X = tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(
        1, 1), activation='tanh', input_shape=input_shape, padding='same')(X_input)

    # Add average pooling layer with pool size of (2, 2) and stride of 2
    X = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2), strides=2, padding='valid')(X)

    # Add convolutional layer with 16 filters, kernel size of (5, 5), stride of (1, 1), and hyperbolic tangent
    # activation
    X = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(
        1, 1), activation='tanh', padding='valid')(X)

    # Add average pooling layer with pool size of (2, 2) and stride of 2
    X = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2), strides=2, padding='valid')(X)

    # Add convolutional layer with 120 filters, kernel size of (5, 5), stride of (1, 1), and hyperbolic tangent
    # activation
    X = tf.keras.layers.Conv2D(120, kernel_size=(5, 5), strides=(
        1, 1), activation='tanh', padding='valid')(X)

    # Flatten tensor
    X = tf.keras.layers.Flatten()(X)

    # Add fully-connected dense layer with 84 units and hyperbolic tangent activation
    X = tf.keras.layers.Dense(84, activation='tanh')(X)

    # Add final dense layer with single unit and no activation function
    X = tf.keras.layers.Dense(1)(X)

    # Create model instance with input tensor and output tensor
    model = tf.keras.Model(inputs=X_input, outputs=X, name='leNet')

    return model


# In[ ]:


lenet_model = LeNet_Model((Input_Size, Input_Size, 3))

# Count number of parameters in the model
num_params = lenet_model.count_params()
print(f'Number of parameters: {num_params:,}')

# Print information about available memory
print_memory_info()

# Print a summary of the model
lenet_model.summary()

# In[ ]:


# Create a callback that will stop training when the validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Use the Adam optimization algorithm with a learning rate of 0.01
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# Compile the model with the Adam optimizer, mean squared error loss, and mean absolute error metric
lenet_model.compile(optimizer=adam_optimizer, loss=tf.keras.losses.MeanAbsoluteError(),
                    metrics=tf.keras.metrics.mean_absolute_error)

# Train the model using the early stopping callback and using multiprocessing with available CPU's
hist = lenet_model.fit(train_dataset, epochs=200,
                       validation_data=validation_dataset,
                       callbacks=[early_stopping],
                       use_multiprocessing=True,
                       workers=os.cpu_count()
                       )

# Save the model and training history in HDF5 and npy format, respectively
tf.keras.models.save_model(lenet_model, './denememodel/cnn_model.h5')
np.save('./denememodel/cnn_history.npy', hist.history)

# Uncomment the following lines to restore the model from the HDF5 file and retrieve the training history
# model = tf.keras.models.load_model('./denememodel/model.h5')
# history1=np.load('./denememodel/history1.npy',allow_pickle='TRUE').item()


print("\n")
print("-----------------------------------------------")

# Print information about available memory
print_memory_info()

# In[ ]:


# Evaluate the model
test_loss = lenet_model.evaluate(test_dataset, workers=-1)
print('Test loss:', test_loss)

print("\n")
print("-----------------------------------------------")

hist_graphs(hist)


# ### CNN Model

# In[28]:


def Conv(X, conv_feature_maps=4, conv_kernel=(3, 3), conv_strides=(1, 1),
         conv_padding='same', activation='relu',
         name=None):
    """
    Help function for convolutional + max pooling layers

    Arguments:
    X -- input tensor

    Returns:
    model -- a Model() instance in TensorFlow
    """
    # CONV -> Batch Normalization -> ReLU Block applied to X (3 lines of code)
    X = tf.keras.layers.Conv2D(conv_feature_maps, conv_kernel,
                               strides=conv_strides, padding=conv_padding, activation=None, name=name)(X)
    X = tf.keras.layers.BatchNormalization(axis=-1)(X)
    X = tf.keras.layers.Activation(activation)(X)

    return X


# In[29]:


def ConvPool(X, conv_feature_maps=4, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', activation='relu',
             pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same', name=None):
    """
    Help function for convolutional + max pooling layers

    Arguments:
    X -- input tensor

    Returns:
    model -- a Model() instance in TensorFlow
    """

    # CONV -> Batch Normalization -> ReLU Block applied to X (3 lines of code)
    X = tf.keras.layers.Conv2D(conv_feature_maps, conv_kernel,
                               strides=conv_strides, padding=conv_padding, activation=None, name=name)(X)
    X = tf.keras.layers.BatchNormalization(axis=-1)(X)
    X = tf.keras.layers.Activation(activation)(X)

    # MAXPOOL (1 line of code)
    X = tf.keras.layers.MaxPooling2D(
        pool_size, pool_strides, padding=pool_padding)(X)

    return X


# In[30]:


# FUNCTION: SignModel

def CNN_Model(input_shape):
    """
    Implementation of the CNN Model

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in TensorFlow
    """

    # START CODE HERE ### (1 line of code for each instruction)
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = tf.keras.Input(input_shape)

    # FIRST CONV + MAXPOOL BLOCK
    X = ConvPool(X_input, conv_feature_maps=32, conv_kernel=(11, 11), conv_strides=(1, 1),
                 conv_padding='same', activation='relu',
                 pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same', name="Conv1")

    X = ConvPool(X, conv_feature_maps=64, conv_kernel=(5, 5), conv_strides=(1, 1),
                 conv_padding='same', activation='relu',
                 pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same', name="Conv2")

    X = ConvPool(X, conv_feature_maps=128, conv_kernel=(5, 5), conv_strides=(1, 1),
                 conv_padding='same', activation='relu',
                 pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same', name="Conv3")

    X = ConvPool(X, conv_feature_maps=256, conv_kernel=(5, 5), conv_strides=(1, 1),
                 conv_padding='same', activation='relu',
                 pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same', name="Conv4")

    # FLATTEN THE TENSOR
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dropout(0.4)(X)

    X = tf.keras.layers.Dense(256, activation='relu', name="Dense1")(X)
    X = tf.keras.layers.Dropout(0.4)(X)
    X = tf.keras.layers.Dense(128, activation='relu', name="Dense2")(X)
    X = tf.keras.layers.Dropout(0.4)(X)
    X = tf.keras.layers.Dense(1, activation=None, name="Output_Layer")(X)

    # Create model
    model = tf.keras.Model(inputs=X_input, outputs=X, name='CNN_Model')

    return model


# In[31]:

cnn_model = CNN_Model((Input_Size, Input_Size, 3))

# In[32]:


num_params = cnn_model.count_params()
print(f'Number of parameters: {num_params:,}')

print_memory_info()

cnn_model.summary()

# number of parameters in CNN model : # feature maps this layer * kernel size * # feature maps previous layer + #
# bias (feature maps this layer)


# In[ ]:


# Create a callback that will interrupt training when the validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# Compile the model
cnn_model.compile(optimizer=adam_optimizer, loss=tf.keras.losses.MeanAbsoluteError())

# Train the model
hist = cnn_model.fit(train_dataset, epochs=200, validation_data=validation_dataset,
                     callbacks=[early_stopping], use_multiprocessing=True, workers=4
                     )

# Save the model in HDF5 format
tf.keras.models.save_model(cnn_model, './denememodel/cnn_model.h5')
np.save('./denememodel/cnn_history.npy', hist.history)

# Restore the model from the HDF5 file
# model = tf.keras.models.load_model('./denememodel/model.h5')
# Get back the history
# history1=np.load('./denememodel/history1.npy',allow_pickle='TRUE').item()

print("\n")
print("-----------------------------------------------")
# Evaluate the model
test_loss, test_mae = cnn_model.evaluate(test_dataset, workers=-1)
print('Test loss:', test_loss)
print('Test mae:', test_mae)

print("\n")
print("-----------------------------------------------")

print_memory_info()

# In[ ]:


hist_graphs(hist)

# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# Show the structure of the model through building blocks
tf.keras.utils.plot_model(dnn_model, to_file='./denememodel/dnn_model.png')

from IPython.display import Image

Image("./denememodel/dnn_model.png")
