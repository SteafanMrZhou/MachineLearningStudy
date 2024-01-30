# @Author   : Steafan
# @Desc     : 
# @File     : DataSource.py
# @Create   : 2024-01-30 15:48

import os
import tensorflow as tf
from tensorflow import keras
from keras import datasets

class DataSource(object):
    def __init__(self):
        # data_path = os.path.abspath(os.path.dirname(__file__)) + '/../data_set_tf2/mnist.npz'
        # (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path=data_path)
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
        train_images, test_images = train_images / 255.0, test_images / 255.0

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels

DataSource()