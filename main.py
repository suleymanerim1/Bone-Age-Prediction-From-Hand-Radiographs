# Import Packages
import datetime
import os

import numpy as np
import tensorflow as tf
from IPython.display import Image
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow import keras

import utils
import model_trials

#Create dataset paths
train_path = 'Bone Age Datasets\\train'
train_csv_path = 'Bone Age Datasets\\train.csv'

validation_path = 'Bone Age Datasets\\validation'
validation_csv_path = 'Bone Age Datasets\\validation.csv'

test_path = 'Bone Age Datasets\\test'
test_csv_path = 'Bone Age Datasets\\test.csv'

Input_Size = 224

train = (train_path, train_csv_path)
val = (validation_path, validation_csv_path)
test = (test_path, test_csv_path)
data_paths = [train, val, test]
datasets = []
len_dataset = []

# Read images and information from paths
for path in data_paths:
    # return images paths list and features (id,male,age)
    image_file_list, features = utils.image_csv_match(path[0], path[1])

    # create a tf.dataset of test images
    images_dataset = utils.image_dataset_creator_from_path(image_file_list, Input_Size)

    # get age from features
    age = (features[:, -1]).astype(float)
    # Create an age tf.dataset from the NumPy array
    age_dataset = tf.data.Dataset.from_tensor_slices(age)

    # get gender from features
    gender = (features[:, 1]).astype(int)
    # Create an age tf.dataset from the NumPy array
    gender_dataset = tf.data.Dataset.from_tensor_slices(gender)

    # Create a dataset of images zipped with age
    datasets.append(tf.data.Dataset.zip(((images_dataset, gender_dataset), age_dataset)))
    #datasets.append(tf.data.Dataset.zip((images_dataset, age_dataset)))

    len_dataset.append(len(gender))

# Create Datasets
train_dataset = datasets[0]
val_dataset = datasets[1]
test_dataset = datasets[2]

train_len = len_dataset[0]
val_len = len_dataset[1]
test_len = len_dataset[2]

batch_size = 64
train_dataset = utils.create_dataset(train_dataset,
                                     batch_size=batch_size,
                                     shuffle=False,
                                     augment=True,
                                     cache_file='train_cache'
                                     )
val_dataset = utils.create_dataset(val_dataset,
                                   batch_size=batch_size,
                                   cache_file='validation_cache'
                                   )
test_dataset = utils.create_dataset(test_dataset,
                                    batch_size=batch_size,
                                    cache_file='test_cache')

number_train_repeat = 2
train_steps = int(np.ceil(train_len / batch_size))*number_train_repeat
val_steps = int(np.ceil(val_len / batch_size))*number_train_repeat
test_steps = int(np.ceil(test_len / batch_size))*number_train_repeat

print(train_dataset)
print(val_dataset)
print(test_dataset)

tf.keras.backend.clear_session()
model = model_trials.model5_bn_gn((Input_Size, Input_Size, 1))
#model = model_trials.model20()

num_params = model.count_params()
print(f'Number of parameters: {num_params:,}\n')

model.summary()

#Callbacks
# Create a callback that will interrupt training when the validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, restore_best_weights=True)

# tensorboard
log_dir = "./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# learning rate schedule
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=4, min_lr=0.001, verbose=1)


# Compile and Fit
initial_lrate = 0.1
#initial_lrate_for_transfer_learning = 0.001
adam_optimizer = keras.optimizers.Adam(learning_rate=initial_lrate)
# Compile the model
model.compile(optimizer=adam_optimizer, loss=tf.keras.losses.MeanAbsoluteError(),
              metrics=tf.keras.metrics.mean_absolute_error)

# Train the model
hist = model.fit(train_dataset, epochs=50,
                 steps_per_epoch=train_steps,
                 validation_data=val_dataset,
                 validation_steps=val_steps,
                 callbacks=[reduce_lr, early_stopping, tensorboard_callback],
                 use_multiprocessing=True,
                 workers=os.cpu_count()
                 )

# Evaluate the model
test_mae = model.evaluate(test_dataset, steps=test_steps, workers=os.cpu_count(),use_multiprocessing = True)
print('Test loss:', test_mae)

print("\n")
print("-----------------------------------------------")

utils.hist_graphs(hist)


path = './denememodel/'
path_real = './my_models/'
# Save the model in HDF5 format
keras.models.save_model(model, filepath=path + 'dummymodel.h5')
# save the history
np.save(file=path + 'dummyhistory.npy', arr=hist.history)
# save best epoch
np.save(file=path + 'best_epoch.npy', arr=early_stopping.best_epoch)
# Show the structure of the model through building blocks
keras.utils.plot_model(model, to_file=path + 'dummymodel.png', show_shapes=True)
Image(path + 'dummymodel.png')

# Restore the model from the HDF5 file
# model = tf.keras.models.load_model(path + 'model.h5')
# Get back the history
# history1=np.load(file = path + 'history1.npy',allow_pickle='TRUE').item()
# Get back the best epoch
# best_epoch = np.load(file = path + 'best_epoch.npy',allow_pickle='TRUE').item()


# number of parameters
# model.count_params()

# tensorboard --logdir C:\Users\erim_\PycharmProjects\HDAProject\logs\fit