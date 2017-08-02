import keras
import sys
sys.path.append('kernel')
import DigitRecognizer; reload(DigitRecognizer)
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
datapath='../data/mnist/'

#Get model and get batches as numpy array (28x28) pixels and labels (0..9)
model = DigitRecognizer.DigitRecognizer()
training_data,training_labels = model.get_training_data(datapath + 'train_no_header.csv')
training_labels = keras.utils.to_categorical(training_labels, num_classes=10)
validation_batches = model.get_training_data(datapath + 'validation.csv')

#model.fit(training_data, validation_data, nb_epoch=1)
#model.fit(training_data, training_labels, nb_epoch=10,validation_split=.1)

#np.set_printoptions(threshold='nan')
#print(training_data[1:100,...])
#print("\nlabels\n")
#print(training_labels[1:100,...])
model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(training_data, training_labels, batch_size=32, epochs=10, validation_split=.1)
