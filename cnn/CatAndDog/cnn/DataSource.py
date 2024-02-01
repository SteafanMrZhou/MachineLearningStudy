# @Author   : Steafan
# @Desc     : 猫狗识别数据源处理
# @File     : DataSource.py
# @Create   : 2024-02-01 15:03

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib as path
import matplotlib.pyplot as plt


class DataSource(object):

    def __init__(self):
        batch_size = 100
        img_height, img_width = 64, 64
        img_data_dir = path.Path("../data/")

        print("loading images...")
        train_data = keras.utils.image_dataset_from_directory(
            img_data_dir,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode='rgb',
            validation_split=0.2,
            subset='training',
            seed=123
        )
        validation_data = keras.utils.image_dataset_from_directory(
            img_data_dir,
            image_size=(img_height, img_width),
            batch_size=batch_size,
            color_mode='rgb',
            validation_split=0.2,
            subset='validation',
            seed=123
        )
        print("finished!")

        # class_names = train_data.class_names

        self.train_data = train_data
        self.validation_data = validation_data
        # self.train_labels = class_names

        train_images = []
        # 1-dog,0-cat
        train_labels = []
        for images, labels in train_data.as_numpy_iterator():
            for i in range(len(images)):
                train_images.append(images[i] / 255.0)
                train_labels.append(labels[i])

        validation_images = []
        validation_labels = []
        for images, labels in validation_data.as_numpy_iterator():
            for i in range(len(images)):
                validation_images.append(images[i] / 255.0)
                validation_labels.append(labels[i])

        train_images = np.array(train_images).reshape((16000, 64, 64, 3))
        # train_labels = np.array(train_labels).reshape((16000, 64, 64, 3))
        self.train_images = train_images
        self.train_labels = np.array(train_labels)
        # validation_images = np.array(validation_images).reshape((16000, 64, 64, 3))
        self.validation_images = validation_images
        self.validation_labels = validation_labels
        print(train_images[0].shape)

        # plt.figure(figsize=(10, 10))
        #
        # for images, labels in train_data.take(1):
        #     for i in range(9):
        #         ax = plt.subplot(3, 3, i + 1)
        #         plt.imshow(images[i].numpy().astype("uint8"))
        #         plt.title(class_names[labels[i]])
        #         plt.axis("off")
        #
        # plt.show()


    def cnnModels(self):
        models = keras.models.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            # keras.layers.Conv2D(128, (3, 3), activation='relu'),
            # keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(1, activation='softmax')
        ])

        models.summary()

        self.model = models

    def training(self):
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics="accuracy"
        )

        # print(type(self.train_images))
        # print(type(self.train_labels))

        self.model.fit(
            self.train_images,
            self.train_labels,
            validation_data=self.validation_data,
            epochs=5
        )


if __name__ == "__main__":
    ds = DataSource()
    ds.cnnModels()
    ds.training()

