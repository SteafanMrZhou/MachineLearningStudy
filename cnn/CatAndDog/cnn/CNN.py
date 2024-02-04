# @Author   : Steafan
# @Desc     : 猫狗识别CNN实现(暂时废弃)
# @File     : CNN.py
# @Create   : 2024-02-01 15:44

import tensorflow as tf
from tensorflow import keras
import numpy as np
from cnn.DataSource import DataSource

class CNN(object):

    def __init__(self):
        models = keras.models.Sequential([
            keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1, activation='softmax')
        ])

        models.summary()

        self.model = models
        self.data = DataSource()



    def training(self):
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics="accuracy"
        )

        train_images = self.data.train_images
        train_labels = self.data.train_labels
        # train_images = np.array(list(train_images)).reshape((64, 64, 3))

        print(train_images[0].shape)

        # self.model.fit(
        #     self.data.train_images,
        #     self.data.train_labels,
        #     epochs=10
        # )

if __name__ == '__main__':
    cnn = CNN()
    cnn.training()