import csv
import os

from keras.models import Sequential
from keras.layers import *

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data_generator import data_generate

# Parameters
batch_sizes = 32
dropout = 0.3
epochs = 10
num_of_datafiles = 5

resource_dir = '/home/kx/training/'

# List all folders that contain the training data
dirs = []

# Load all driving log csv files from respective dirs
lines = []

for i in range(1, num_of_datafiles + 1):
	for item in os.listdir(resource_dir + str(i) + '/IMG'):
		dirs.append(item)

	csvfile =  open(resource_dir + str(i) +'/driving_log.csv')
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# Splitting train/validation sets
X_train, X_valid = train_test_split(lines, test_size=0.2)

# Creating generators for each dataset
X_train_data = data_generate(X_train, batch_sizes)
X_valid_data = data_generate(X_valid, batch_sizes)

model = Sequential()

# Cropping images from (160, 320) to (70 ,25)
model.add(Cropping2D(cropping=((70 ,25), (0,0)), input_shape=(160 ,320 ,3)))
# Normalisation
model.add(Lambda(lambda x: x / 255.0 - 0.5))

# NVIDIA End-to-end neural net implementation
model.add(Convolution2D(filters=24, kernel_size=5, strides=(2 ,2), activation='relu'))
model.add(Convolution2D(filters=36, kernel_size=5, strides=(2 ,2),activation='relu'))
model.add(Convolution2D(filters=48, kernel_size=5, strides=(2 ,2),activation='relu'))
model.add(Convolution2D(filters=64, kernel_size=3, activation='relu'))
model.add(Convolution2D(filters=64, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(dropout))
model.add(Dense(50))
model.add(Dropout(dropout))
model.add(Dense(10))
model.add(Dense(1, activation='linear'))

print('Total number of samples: ', len(lines)*2)

# Loss function and optimiser step
model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(X_train_data, steps_per_epoch=len(X_train)/batch_sizes, validation_data=X_valid_data, validation_steps=len(X_valid)/batch_sizes, epochs=epochs, verbose=1)
print('Training complete!')
model.save('model.h5')
print('Model saved!')

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
