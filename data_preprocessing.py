# student name: Hien Le
# student number: 101044264


import numpy as np
import glob
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tqdm import tqdm
import pickle
import math


class LoadData():

    def __init__(self, img_size=224):
        self.img_size = img_size

    def load_images(self, label, img_width, img_height):
        img_paths = glob.glob(f'./dataset/{label}/*.jpg')

        print(f'\nLoading and Resizing {label} images...')
        images = [load_img(path, target_size=(img_width, img_height))
                  for path in tqdm(img_paths)]

        print('Converting Image to Array...')
        img_arr = np.array([img_to_array(i)/255 for i in tqdm(images)])
        
        labels = np.repeat(label, img_arr.shape[0])

        return img_arr, labels

    def load(self, show_plot=True, save_pkl=False):
        classes = ['daisy', 'rose', 'sunflower', 'dandelion', 'tulip']
        flower_X = []
        flower_Y = []

        for flower in classes:
            img, label = self.load_images(flower, self.img_size, self.img_size)
            flower_X.append(img)
            flower_Y.append(label)

        self.X = np.concatenate(flower_X, axis=0)
        self.Y = np.concatenate(flower_Y, axis=0)

        if show_plot:
            self.plot_stat()

        if save_pkl:
            self.export_pickle()

        return self.X, self.Y

    def plot_stat(self):
        unique, counts = np.unique(self.Y, return_counts=True)
        x_plot = list(range(len(unique)))
        plt.bar(x_plot, counts, color='blue')
        plt.xticks(x_plot, unique)
        plt.xlabel('Classes')
        plt.ylabel('Counts')
        plt.title('Number of samples per classes')
        plt.savefig('./results/class-counts.png')
        plt.show()

    def display_samples(self, X, Y, classes, n_images=12, offset=555):
        fig, axarr = plt.subplots(2, math.ceil(n_images/2), figsize=(8, 8))

        for i in range(n_images):
            j = i//2
            k = i%2
            flower_label = np.argmax(Y[offset+i])
            axarr[k,j].imshow(X[offset+i])
            axarr[k,j].set_title(classes[flower_label])
            axarr[k,j].axis('off')

        # plt.tight_layout()
        plt.show()

    def export_pickle(self):
        with open('./dataset/flowers.pkl', 'wb') as f:
            pickle.dump(self.X, f)
        with open('./dataset/flowers-labels.pkl', 'wb') as f:
            pickle.dump(self.Y, f)
