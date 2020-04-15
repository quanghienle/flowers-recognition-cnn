# student name: Hien Le
# student number: 101044264

from skimage.util import random_noise
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import os
import logging


from data_preprocessing import LoadData
from CNN import CNN


import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel(logging.ERROR)


# Adding noise to the dataset
def add_noise(img):
    noise_level = random.uniform(0, 0.2)
    return random_noise(img, mode='s&p', amount=noise_level)

noise_gen = ImageDataGenerator(preprocessing_function=add_noise)


# Image augmentation
aug_gen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.1,  
    shear_range=0.2, 
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True) 



if __name__ == '__main__':

    loader = LoadData(img_size=224)
    X, labels = loader.load(show_plot=False, save_pkl=False)
    classes, index_Y = np.unique(labels, return_inverse=True)

    Y = to_categorical(index_Y).astype('float32')

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=99)

    input_shape = X[0].shape
    output_size = len(classes)

    while(True):
        print('\n')
        print('\t[1]. Display samples.')
        print('\t[2]. Display class counts.')
        print('\t[3]. Run CNN model with Original data.')
        print('\t[4]. Run CNN model with Noisy data.')
        print('\t[5]. Run CNN model with Augmented data.')
        print('\t[q]. Quit.')
        
        choice = input('Enter an option: ')
        model = None

        if choice == '1':
            loader.display_samples(X_train, Y_train, classes, n_images=14)
        elif choice == '2':
            loader.plot_stat()
        elif choice == '3':
            model = CNN(input_shape, output_size, generator=None)
        elif choice == '4':    
            model = CNN(input_shape, output_size, generator=noise_gen)
        elif choice == '5':
            model = CNN(input_shape, output_size, generator=aug_gen)
        elif choice == 'q':
            break

        if model:
            model.train(X_train, Y_train, X_test, Y_test)
