# Thomas Ehret
# CS519 Applied ML
# Dr Cao
# NMSU Sp23
# project 5

# this code is based on the following tutorial:
# keras api docuemtnation: https://keras.io/api/
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

# Load preprocessed dataset
X = np.load('dataset_images.npy')
y = np.load('dataset_labels.npy')

# Define learning rate schedule
def lr_schedule(epoch):
    lr = 0.001
    if epoch > 5:
        lr *= 0.1
    return lr

# Define CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
# model.fit(X, y, epochs=10, validation_split=0.2)

# Train model
history = model.fit(X, y, epochs=10, validation_split=0.2)

# Plot accuracy and learning rate over epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

lr = [lr_schedule(epoch) for epoch in range(len(history.history['accuracy']))]
plt.plot(lr)
plt.title('Learning Rate')
plt.ylabel('Learning Rate')
plt.xlabel('Epoch')
plt.show()


# Save model
model.save('cnn_model.h5')
