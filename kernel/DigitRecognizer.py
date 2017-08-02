from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D
from keras.layers import Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np

K.set_image_dim_ordering('th')

class DigitRecognizer():
    """
        Toy model to test our cnn models
        in preparation for whale recognition
    """

    def __init__(self):
        #here I need to create the layers
        self.__create_model()

    def __create_model(self):
        """
            Creates the layers of the model
        """
        self.model = Sequential()
        self.model.add(Conv2D(7, (3,3), input_shape=(1,28,28), activation='relu'))
        self.model.add(Conv2D(7, (3,3), activation='relu')) 
        self.model.add(Flatten())
        self.model.add(Dense(4024, activation='relu' ))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()

    def get_training_data(self, path):
        """
            From the path to a directory generates batches of images that can be passed to the fit function.
        """
        f = open(path, "r")
        #TODO fix this
        count = 0
        for l in f:
            count = count + 1

        f = open(path, "r")
        x = np.zeros((count,1,28,28),dtype=np.float32)
        y = np.zeros((count,1),dtype=np.float32)

        index = 0
        for l in f:
            x[index], y[index] = self._training_line_to_numpy(l)  
            index = index + 1

        return (x,y)


    def get_test_data(self, path):
        """
            From the path generate batches of images data that can be passed to the predict function
        """
        f = open(path, "r")
        count = 0
        for l in f:
            count = count + 1

        f = open(path, "r")
        x = np.zeros((count,1,28,28),dtype=np.float32)

        index = 0
        for l in f:
            x[index] = self._test_line_to_numpy(l)  
            index = index + 1

        return x

 

    def fit(self, x, y, nb_epoch=1, validation_split=0.0):
        """
            Fits the model on the provided data on a batch by batch basis.
        """
        self.model.fit(x, y, epochs=nb_epoch, validation_split=validation_split)

    def predict(self, data, batch_size=8):
        """
            Uses the trained model to predict classes for test data
        """
        return self.model.predict(data, batch_size=32);

    def _training_line_to_numpy(self,line):
        values = line.split(',')
        clss = values[0]
        data = values[1:]
        pixels = np.zeros((1,28,28),dtype=np.float32)

        if len(data) == 28 * 28:
            for i in range(0,28):
                for j in range(0,28):
                    try:
                        pixels[0,i,j] = float(data[(i * 28) + j])
                    except:
                        x = 1
                        #print(len(data), i, j)
                        #print(data[(i*28)+j])
        else:
            print("error bad data")
        return (pixels, float(clss))

    #This should be integrated with _training_line_to_numpy to reduce duplication
    def _test_line_to_numpy(self,line):
        values = line.split(',')
        pixels = np.zeros((1,28,28),dtype=np.float32)
    
        if len(values) == 28 * 28:
            for i in range(0,28):
                for j in range(0,28):
                    try:
                        pixels[0,i,j] = float(values[(i * 28) + j])
                    except:
                        x = 1
        else:
            print("error bad data")
        return pixels
        



