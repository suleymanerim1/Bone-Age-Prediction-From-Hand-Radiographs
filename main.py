# Import Packages
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import os
import utils

Input_Size = 224

# Create train dataset
train_path = 'Bone Age Datasets\\train'
train_csv_path = 'Bone Age Datasets\\train.csv'

validation_path = 'Bone Age Datasets\\validation'
validation_csv_path = 'Bone Age Datasets\\validation.csv'

test_path = 'Bone Age Datasets\\test'
test_csv_path = 'Bone Age Datasets\\test.csv'

train = (train_path,train_csv_path)
val = (validation_path,validation_csv_path)
test = (test_path,test_csv_path)
data_paths = [train,val,test]
datasets = []
for path in data_paths:
    # return images paths list and features (id,male,age)
    image_file_list, features = utils.image_csv_match(path[0], path[1])

    # create a tf.dataset of test images
    images_dataset = utils.image_dataset_creator_from_path(image_file_list)

    # get age from features
    age = (features[:, -1]).astype(float)

    # Create an age tf.dataset from the NumPy array
    age_dataset = tf.data.Dataset.from_tensor_slices(age)

    # Create a dataset of images zipped with age
    datasets.append( tf.data.Dataset.zip((images_dataset, age_dataset)))

train_dataset = datasets[0]
val_dataset = datasets[1]
test_dataset = datasets[2]

