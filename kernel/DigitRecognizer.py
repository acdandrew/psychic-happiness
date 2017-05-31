from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

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
        self.model.add(Conv2D(7, 3, input_shape=(28,28,1), activation='relu'))
        self.model.add(Conv2D(7, 3, activation='relu')) 
        self.model.add(Dense(4096, activation='tanh'))
        self.model.add(Dense(10, activation='tanh'))
        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

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
        x = np.zeros((count,28,28,1))
        y = np.zeros((count,1))

        index = 0
        for l in f:
            x[index], y[index] = self._training_line_to_numpy(l)  

        return (x,y)


    def get_test_data(self, path):
        """
            From the path generate batches of images data that can be passed to the predict function
        """

    def fit(self, batches, val_batches, nb_epoch=1):
        """
            Fits the model on the provided data on a batch by batch basis.
        """

    def predict(self, batches, batch_size=8):
        """
            Uses the trained model to predict classes for test data
        """

    def _training_line_to_numpy(self,line):
        values = line.split(',')
        clss = values[0]
        data = values[1:]
        pixels = np.zeros((28,28,1))

        if len(data) == 28 * 28:
            for i in range(0,28):
                for j in range(0,28):
                    try:
                        pixels[i,j,0] = float(data[(i * 28) + j])
                    except:
                        print(len(data), i, j)
                        print(data[(i*28)+j])
        else:
            print("error bad data")
        return (pixels, clss)


