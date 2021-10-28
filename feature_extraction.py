from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.applications.vgg16 import VGG16

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
#
train_dir = r"./dataset/train"
valid_dir = r"./dataset/test"

img_width, img_height = 224, 224  # Default input size for VGG16
#
# Instantiate convolutional base

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(img_width, img_height, 3))

# Show architecture
conv_base.summary()
# Extract features
import os, shutil
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 16


def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count, 4))
    # Preprocess data
    generator = datagen.flow_from_directory(directory,
                                            target_size=(img_width, img_height),
                                            batch_size=batch_size,
                                            class_mode='categorical')
    # Pass data through convolutional base
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


train_features, train_labels = extract_features(train_dir, 112)  # Agree with our small dataset size
validation_features, validation_labels = extract_features(valid_dir, 48)
# test_features, test_labels = extract_features(test_dir, test_size)
train_labels
# print(len(train_labels))
epochs = 2000

model = Sequential()
model.add(GlobalAveragePooling2D(input_shape=(10, 10, 512)))
model.add(Dense(4, activation='softmax'))
model.summary()
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_loss',
                             save_best_only=True, mode='auto')

# Compile model
from keras.optimizers import Adam

model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['acc'])

# Train model
history = model.fit(train_features, train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[checkpoint],
                    validation_data=(validation_features, validation_labels))
# Plot results
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

from keras.preprocessing import image


def prediction(img_path):
    org_img = image.load_img(img_path)
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range
    img_tensor /= 255.  # Normalize to [0,1] for plt.imshow application
    plt.imshow(org_img)
    plt.axis('on')
    plt.show()

    # Extract features
    features = conv_base.predict(img_tensor.reshape(1, img_width, img_height, 3))

    # Make prediction
    # pred_dir = r"/archive/seg_pred/seg_pred/"
    try:
        prediction = model.predict(features)
    except:
        prediction = model.predict(features.reshape(1, 10 * 10 * 512))

    classes = ["African - American - Face", "Asian - Face", "Asian - Middle-Eastern - Face", "White - Face"]
    print("I see..." + str(classes[np.argmax(np.array(prediction[0]))]))


# import random
# pred_files = random.sample(os.listdir(pred_dir), 10)
# for f in pred_files:
# prediction(pred_dir + f)
path = "dataset/pred/"
onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
print(onlyfiles)
for i in onlyfiles:
    prediction(path + i)
