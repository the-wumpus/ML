# Thomas Ehret
# CS519 Applied ML
# Dr Cao
# NMSU Sp23
# project 5

import tensorflow as tf
from keras.utils import to_categorical
import numpy as np

# This code is based on the Keras documentation for image_dataset_from_directory
# Define image path and label mapping
image_path = 'images'
label_mapping = {'strep_neg': 0, 'strep_pos': 1}

# Define dataset parameters
image_size = (224, 224)
batch_size = 32

# Load images and labels using image_dataset_from_directory
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    image_path,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=123,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
)

# Extract images and labels as NumPy arrays
images = []
labels = []
for batch in dataset:
    batch_images, batch_labels = batch
    for i in range(len(batch_images)):
        images.append(batch_images[i].numpy())
        labels.append(batch_labels[i].numpy())

# Convert labels to binary categories
labels = to_categorical(labels)

# Save preprocessed dataset
np.save('dataset_images.npy', np.array(images))
np.save('dataset_labels.npy', labels)

