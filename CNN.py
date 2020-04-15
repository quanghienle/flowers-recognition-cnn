# student name: Hien Le
# student number: 101044264


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Flatten, LeakyReLU, Input, GaussianNoise


class CNN:

    def __init__(self, input_shape, output_size, generator=None, optimizer='adam'):
        self.generator = generator
        self.model = self.create_model(input_shape, output_size, optimizer)

    def create_model(self, input_shape, output_size, optimizer):

        cnn = Sequential()

        cnn.add(Input(shape=input_shape))

        cnn.add(Conv2D(filters=64, kernel_size=3, padding="same", activation='relu'))
        cnn.add(MaxPooling2D(pool_size=2, strides=2))

        cnn.add(Conv2D(filters=128, kernel_size=3, padding="same", activation='relu'))
        cnn.add(MaxPooling2D(pool_size=2, strides=2))

        cnn.add(Dropout(0.5))

        cnn.add(Conv2D(filters=256, kernel_size=3, padding="same", activation='relu'))
        cnn.add(Conv2D(filters=256, kernel_size=3, padding="same", activation='relu'))
        cnn.add(MaxPooling2D(pool_size=2, strides=2))

        cnn.add(Conv2D(filters=512, kernel_size=3, padding="same", activation='relu'))
        cnn.add(Conv2D(filters=512, kernel_size=3, padding="same", activation='relu'))
        cnn.add(MaxPooling2D(pool_size=2, strides=2))

        cnn.add(Dropout(0.5))

        cnn.add(Conv2D(filters=512, kernel_size=3, padding="same", activation='relu'))
        cnn.add(Conv2D(filters=512, kernel_size=3, padding="same", activation='relu'))
        cnn.add(MaxPooling2D(pool_size=2, strides=2))

        cnn.add(GlobalAveragePooling2D())

        cnn.add(Dense(output_size, activation='softmax'))

        cnn.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return cnn

    def train(self, X_train, Y_train, X_test, Y_test, batch_size=32, epochs=100):

        if self.generator:
            history = self.model.fit_generator(self.generator.flow(X_train, Y_train, batch_size=batch_size),
                                               steps_per_epoch=len(X_train) // batch_size,
                                               epochs=epochs,
                                               validation_data=(X_test, Y_test))
        else:
            history = self.model.fit(X_train, Y_train, epochs=100,
                                     validation_data=(X_test, Y_test),
                                     batch_size=batch_size)

        self.plot_result(history, 'loss')
        self.plot_result(history, 'accuracy')
        return history

    def plot_result(self, history, type='loss'):
        y1 = history.history[type]
        y2 = history.history[f'val_{type}']

        x_plot = list(range(len(y1)))
        plt.plot(x_plot, y1, label=f'training {type}')
        plt.plot(x_plot, y2, label=f'testing {type}')
        plt.legend()
        plt.show()
