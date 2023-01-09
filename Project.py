#!/usr/bin/env python
# coding: utf-8

#Import Packages
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os

# Avoid Out of Memory errors by setting GPU memory consumption growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

tf.config.list_physical_devices('GPU')

#Load Datasets
def image_csv_match(image_path, csv_path):
    """
    Matches images in the specified directory with their corresponding features stored in a CSV file.

    Parameters:
        image_path (str): The directory containing the images.
        csv_path (str): The path to the CSV file containing the features.

    Returns:
        A tuple containing two lists:
            - A list of image file paths
            - A list of corresponding features
    """

    # Load the filenames of the images
    filenames = os.listdir(image_path)

    # Load the CSV file
    csv = np.loadtxt(csv_path, dtype=str, delimiter=",")

    # Get the list of IDs in the CSV file
    ids = csv[:, 0]

    # Initialize an empty array to store the images and the features
    image_file_list = []
    features = []

    # Iterate over the image filenames
    for filename in filenames:
        # Extract the ID from the filename
        id = filename.split(".")[0]

        # Check if the ID is in the CSV file
        if np.isin(id, ids):
            # Load the image
            file_path = os.path.join(image_path, filename)

            # Extract the corresponding features from the CSV file
            feature = csv[np.where(ids == id), 0:]

            # Append the image and the features to the array
            image_file_list.append(file_path)
            features.append(feature)

    # Convert the array to a NumPy array

    features = np.array(features).squeeze()

    return image_file_list, features


# In[3]:


Input_Size = 224


# In[4]:


def image_dataset_creator_from_path(image_file_list):
    """
    Creates a dataset from a list of image file paths. The images are read, decoded, resized, and standardized.

    Parameters:
        image_file_list (list): A list of image file paths.

    Returns:
        A TensorFlow dataset containing the images.
    """

    # Create a dataset from the list of image paths
    dataset = tf.data.Dataset.from_tensor_slices(image_file_list)

    # Define a function to read and decode an image

    def read_and_decode_image(image_path):
        # Read the image file
        image_string = tf.io.read_file(image_path)
        # Decode the image
        image = tf.image.decode_png(image_string, channels=3)
        # Resize the image
        image_resized = tf.image.resize(image, (Input_Size, Input_Size))
        # Standardized image
        image_standardized = tf.image.convert_image_dtype(
            image_resized, dtype=tf.float32)
        return image_standardized

    # Map the function over the dataset
    dataset = dataset.map(read_and_decode_image)

    return dataset


# ### Create train dataset

# In[5]:


train_path = 'Bone Age Datasets\\train'
train_csv_path = 'Bone Age Datasets\\train.csv'

# return test images paths list and features (id,male,age)
image_file_list, features = image_csv_match(train_path, train_csv_path)

# create a tf.dataset of test images
images_dataset = image_dataset_creator_from_path(image_file_list)

# get age from features
age = (features[:, -1]).astype(float)

# Create an age tf.dataset from the NumPy array
age_dataset = tf.data.Dataset.from_tensor_slices(age)

# Create a dataset of iamges zipped with age
train_dataset = tf.data.Dataset.zip((images_dataset, age_dataset))


# ### Create validation dataset

# In[6]:


validation_path = 'Bone Age Datasets\\validation'
validation_csv_path = 'Bone Age Datasets\\validation.csv'

# return test images paths list and features (id,male,age)
image_file_list, features = image_csv_match(
    validation_path, validation_csv_path)

# create a tf.dataset of test images
images_dataset = image_dataset_creator_from_path(image_file_list)

# get age from features
age = (features[:, -1]).astype(float)

# Create an age tf.dataset from the NumPy array
age_dataset = tf.data.Dataset.from_tensor_slices(age)

# Create a dataset of iamges zipped with age
validation_dataset = tf.data.Dataset.zip((images_dataset, age_dataset))


# ### Create test dataset

# In[7]:


test_path = 'Bone Age Datasets\\test'
test_csv_path = 'Bone Age Datasets\\test.csv'

# return test images paths list and features (id,male,age)
image_file_list, features = image_csv_match(test_path, test_csv_path)

# create a tf.dataset of test images
images_dataset = image_dataset_creator_from_path(image_file_list)

# get age from features
age = (features[:, -1]).astype(float)

# Create an age tf.dataset from the NumPy array
age_dataset = tf.data.Dataset.from_tensor_slices(age)

# Create a dataset of iamges zipped with age
test_dataset = tf.data.Dataset.zip((images_dataset, age_dataset))


# ## Augmentation

# In[8]:


def data_augmentation(images):
    # Randomly flip the images horizontally
    images = tf.image.random_flip_left_right(images)

    # Randomly adjust the brightness of the images
    images = tf.image.random_brightness(images, max_delta=0.5)

    # Randomly adjust the contrast of the images
    images = tf.image.random_contrast(images, lower=0.4, upper=0.8)
    
    # Randomly rotate the images
    #images = tf.image.rot90(images, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

    return images


# ## Create Dataset

# In[9]:


AUTOTUNE = tf.data.AUTOTUNE


def create_dataset(dataset, batch_size, shuffle=False, augment=False, cache_file=None):
    """
    Creates a TensorFlow dataset from a given dataset. The dataset is shuffled, repeated indefinitely, 
    batched, and prefetched as specified. Optionally, the dataset can be augmented and cached.

    Parameters:
        dataset (tf.data.Dataset): The dataset to be transformed.
        batch_size (int): The batch size for the dataset.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        augment (bool, optional): Whether to augment the dataset. Defaults to False.
        cache_file (str, optional): The file path to cache the dataset. Defaults to None.

    Returns:
        A TensorFlow dataset.
    """

    # Cache dataset
    if cache_file:
        dataset = dataset.cache(cache_file)

    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(batch_size*5)

    # Repeat the dataset indefinitely
    dataset = dataset.repeat(2)

    # Batch
    dataset = dataset.batch(batch_size=batch_size)

    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x), y),
                              num_parallel_calls=AUTOTUNE)

    # Prefetch
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


# In[10]:


batch_size = 64
train_dataset = create_dataset(train_dataset,
                               batch_size=batch_size,
                               shuffle=False,
                               augment=False,
                               cache_file='train_cache'
                               )
#
validation_dataset = create_dataset(validation_dataset,
                                    batch_size=batch_size,
                                    cache_file='validation_cache'
                                    )

#

test_dataset = create_dataset(test_dataset,
                              batch_size=batch_size,
                              cache_file='test_cache')


# In[ ]:


del age
del age_dataset
del images_dataset


# ## Visualize  Model Output

# In[11]:


def print_memory_info():
    """
    Prints the current and peak memory usage of the notebook process in megabytes.
    """
    import psutil
    import os

    # Get the current and peak memory usage of the notebook process
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    print("\n")
    print("-----------------------------------------------")
    # Print the current and peak memory usage in megabytes
    print(f"Current memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {memory_info.peak_wset / 1024 / 1024:.2f} MB")
    print("-----------------------------------------------")
    print("\n")


# In[12]:


import matplotlib.pyplot as plt

def hist_graphs(hist):
    # Extract the history data
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
 
    # Set up the subplots
    fig, (ax1) = plt.subplots(1, 1, figsize=(12,4))

    # Plot the loss for the training and validation sets
    ax1.plot(loss)
    ax1.plot(val_loss)
    ax1.set_title('Model loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Training', 'Validation'], loc='upper left')


    plt.show()


# ## Create Models

# ### DNN Model

# In[13]:


# FUNCTION: DNN_Model

def DNN_Model(input_shape):
    """
    Implementation of the DNN_Model
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in TensorFlow
    """
    
    ### START CODE HERE ### (1 line of code for each instruction)
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
    model = tf.keras.Model(inputs = X_input, outputs = X_output, name='DNN_Model1')
    
    ### END CODE HERE ###
    
    return model


# In[14]:


# NETWORK MODEL

### START CODE HERE ### (1 line of code)
dnn_model = DNN_Model((Input_Size, Input_Size, 3))
### END CODE HERE ###


# In[15]:


num_params = dnn_model.count_params()
print(f'Number of parameters: {num_params:,}')

print_memory_info()

dnn_model.summary()


# In[17]:


# Create a callback that will interrupt training when the validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1,restore_best_weights=True)

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# Compile the model
dnn_model.compile(optimizer=adam_optimizer, loss=tf.keras.losses.MeanAbsoluteError(),
                    metrics=tf.keras.metrics.mean_absolute_error)

# Train the model
hist=dnn_model.fit(train_dataset ,epochs=200, 
                     validation_data=validation_dataset,
                      callbacks=[early_stopping], 
                     use_multiprocessing=True,
                     workers=os.cpu_count()
              )


# Save the model in HDF5 format
tf.keras.models.save_model(dnn_model, './denememodel/dnn_model.h5')
np.save('./denememodel/dnn_history.npy',hist.history)

# Restore the model from the HDF5 file
#model = tf.keras.models.load_model('./denememodel/model.h5')
# Get back the history
#history1=np.load('./denememodel/history1.npy',allow_pickle='TRUE').item()


print("\n")
print("-----------------------------------------------")

print_memory_info()


# In[18]:


# Evaluate the model
test_mae = dnn_model.evaluate(test_dataset,workers=-1)
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

    # Add convolutional layer with 16 filters, kernel size of (5, 5), stride of (1, 1), and hyperbolic tangent activation
    X = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(
        1, 1), activation='tanh', padding='valid')(X)

    # Add average pooling layer with pool size of (2, 2) and stride of 2
    X = tf.keras.layers.AveragePooling2D(
        pool_size=(2, 2), strides=2, padding='valid')(X)

    # Add convolutional layer with 120 filters, kernel size of (5, 5), stride of (1, 1), and hyperbolic tangent activation
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
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1,restore_best_weights=True)


# Use the Adam optimization algorithm with a learning rate of 0.01
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


# Compile the model with the Adam optimizer, mean squared error loss, and mean absolute error metric
lenet_model.compile(optimizer=adam_optimizer, loss=tf.keras.losses.MeanAbsoluteError(),
                    metrics=tf.keras.metrics.mean_absolute_error)


# Train the model using the early stopping callback and using multiprocessing with available CPU's
hist=lenet_model.fit(train_dataset ,epochs=200, 
                     validation_data=validation_dataset,
                      callbacks=[early_stopping], 
                     use_multiprocessing=True,
                     workers=os.cpu_count()
              )

# Save the model and training history in HDF5 and npy format, respectively
tf.keras.models.save_model(lenet_model, './denememodel/cnn_model.h5')
np.save('./denememodel/cnn_history.npy',hist.history)



# Uncomment the following lines to restore the model from the HDF5 file and retrieve the training history
# model = tf.keras.models.load_model('./denememodel/model.h5')
# history1=np.load('./denememodel/history1.npy',allow_pickle='TRUE').item()



print("\n")
print("-----------------------------------------------")

# Print information about available memory
print_memory_info()


# In[ ]:


# Evaluate the model
test_loss = lenet_model.evaluate(test_dataset,workers=-1)
print('Test loss:', test_loss)

print("\n")
print("-----------------------------------------------")

hist_graphs(hist)


# ### CNN Model

# In[28]:


def Conv(X, conv_feature_maps=4, conv_kernel=(3, 3), conv_strides=(1, 1), 
             conv_padding='same', activation='relu',
             name = None):
    """
    Help function for convolutional + max pooling layers

    Arguments:
    X -- input tensor

    Returns:
    model -- a Model() instance in TensorFlow
    """
    ### START CODE HERE ###

    # CONV -> Batch Normalization -> ReLU Block applied to X (3 lines of code)
    X = tf.keras.layers.Conv2D(conv_feature_maps, conv_kernel,
                               strides=conv_strides, padding=conv_padding, activation=None,name=name)(X)
    X = tf.keras.layers.BatchNormalization(axis=-1)(X)
    X = tf.keras.layers.Activation(activation)(X)


    ### END CODE HERE ###
    return X


# In[29]:


def ConvPool(X, conv_feature_maps=4, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', activation='relu',
             pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same',name = None):
    """
    Help function for convolutional + max pooling layers

    Arguments:
    X -- input tensor

    Returns:
    model -- a Model() instance in TensorFlow
    """
    ### START CODE HERE ###

    # CONV -> Batch Normalization -> ReLU Block applied to X (3 lines of code)
    X = tf.keras.layers.Conv2D(conv_feature_maps, conv_kernel,
                               strides=conv_strides, padding=conv_padding, activation=None,name=name)(X)
    X = tf.keras.layers.BatchNormalization(axis=-1)(X)
    X = tf.keras.layers.Activation(activation)(X)

    # MAXPOOL (1 line of code)
    X = tf.keras.layers.MaxPooling2D(
        pool_size, pool_strides, padding=pool_padding)(X)

    ### END CODE HERE ###
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
                 pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same',name="Conv1")
    
    X = ConvPool(X, conv_feature_maps=64, conv_kernel=(5, 5), conv_strides=(1, 1),
                 conv_padding='same', activation='relu',
                 pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same',name="Conv2")
        
    X = ConvPool(X, conv_feature_maps=128, conv_kernel=(5, 5), conv_strides=(1, 1),
                 conv_padding='same', activation='relu',
                 pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same',name="Conv3")
                 
    X = ConvPool(X, conv_feature_maps=256, conv_kernel=(5, 5), conv_strides=(1, 1),
                 conv_padding='same', activation='relu',
                 pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same',name="Conv4")                
    

    # FLATTEN THE TENSOR
    X = tf.keras.layers.Flatten()(X)
    X = tf.keras.layers.Dropout(0.4)(X)
    
    X = tf.keras.layers.Dense(256, activation='relu',name="Dense1")(X)
    X = tf.keras.layers.Dropout(0.4)(X)
    X = tf.keras.layers.Dense(128, activation='relu',name="Dense2")(X)
    X = tf.keras.layers.Dropout(0.4)(X)
    X = tf.keras.layers.Dense(1, activation=None,name="Output_Layer")(X)

    # Create model
    model = tf.keras.Model(inputs=X_input, outputs=X, name='CNN_Model')

    ### END CODE HERE ###

    return model


# In[31]:


# NETWORK MODEL

### START CODE HERE ### (1 line of code)
cnn_model = CNN_Model((Input_Size, Input_Size, 3))
### END CODE HERE ###


# In[32]:


num_params = cnn_model.count_params()
print(f'Number of parameters: {num_params:,}')

print_memory_info()

cnn_model.summary()

## number of parameters in CNN model : # feature maps this layer * kernel size * # feature maps previous layer + # bias (feature maps this layer)


# In[ ]:


# Create a callback that will interrupt training when the validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1,restore_best_weights=True)

adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
# Compile the model
cnn_model.compile(optimizer=adam_optimizer, loss=tf.keras.losses.MeanAbsoluteError())

# Train the model
hist=cnn_model.fit(train_dataset ,epochs=200, validation_data=validation_dataset,
              callbacks=[early_stopping], use_multiprocessing=True,workers=4
              )


# Save the model in HDF5 format
tf.keras.models.save_model(cnn_model, './denememodel/cnn_model.h5')
np.save('./denememodel/cnn_history.npy',hist.history)

# Restore the model from the HDF5 file
#model = tf.keras.models.load_model('./denememodel/model.h5')
# Get back the history
#history1=np.load('./denememodel/history1.npy',allow_pickle='TRUE').item()

print("\n")
print("-----------------------------------------------")
# Evaluate the model
test_loss, test_mae = cnn_model.evaluate(test_dataset,workers=-1)
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

