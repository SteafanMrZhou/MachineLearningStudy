# @Author   : Steafan
# @Desc     : 猫狗识别数据源处理+模型实现
# @File     : DataSource.py
# @Create   : 2024-02-01 15:03

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import numpy as np
import pathlib as path
import matplotlib.pyplot as plt


class DataSource(object):

    def __init__(self):
        batch_size = 100
        img_height, img_width = 224, 224
        img_train_data_dir = path.Path("../data/train")
        img_test_data_dir = path.Path("../data/test")

        train_data_generator = ImageDataGenerator(
            validation_split=0.15,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            preprocessing_function=preprocess_input,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        val_data_generator = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.15
        )
        test_data_generator = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )

        train_generator = train_data_generator.flow_from_directory(
            img_train_data_dir,
            target_size=(img_height, img_width),
            shuffle=True,
            seed=123,
            class_mode='categorical',
            batch_size=16,
            subset="training"
        )
        validation_generator = val_data_generator.flow_from_directory(
            img_train_data_dir,
            target_size=(img_height, img_width),
            shuffle=False,
            seed=123,
            class_mode='categorical',
            batch_size=16,
            subset="validation"
        )
        test_generator = test_data_generator.flow_from_directory(
            img_test_data_dir,
            target_size=(img_height, img_width),
            shuffle=False,
            seed=123,
            class_mode='categorical',
            batch_size=16
        )

        nb_train_samples = train_generator.samples
        nb_validation_samples = validation_generator.samples
        nb_test_samples = test_generator.samples
        classes = list(train_generator.class_indices.keys())
        print('Classes:' + str(classes))

        num_classes = len(classes)
        plt.figure(figsize=(15, 15))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            ax.grid(False)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            batch = train_generator.next()
            imgs = (batch[0] + 1) * 127.5
            label = int(batch[1][0][0])
            image = imgs[0].astype('uint8')
            plt.imshow(image)
            plt.title('cat' if label == 1 else 'dog')
        plt.show()

        # 使用 GPU 训练模型
        physical_gpu_device = tf.config.list_physical_devices("GPU")
        # 开启 GPU 内存动态管理（动态增长）
        tf.config.experimental.set_memory_growth(physical_gpu_device[0], True)
        with tf.device("/GPU:0"):
            # 训练模型
            self.cnnModels(img_width, img_height, num_classes=num_classes,
                           train_generator=train_generator,
                           validation_generator=validation_generator,
                           nb_train_samples=nb_train_samples,
                           nb_validation_samples=nb_validation_samples,
                           nb_test_samples=nb_test_samples)

    def cnnModels(self, img_width, img_height, num_classes, train_generator, validation_generator, nb_train_samples,
                  nb_validation_samples, nb_test_samples):
        base_model = InceptionResNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(img_width, img_height, 3)
        )
        x = base_model.output
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(
            100,
            activation='relu'
        )(x)
        predictions = keras.layers.Dense(
            num_classes,
            activation='softmax',
            kernel_initializer='random_uniform'
        )(x)

        model = keras.models.Model(
            inputs=base_model.input,
            outputs=predictions
        )
        for layer in base_model.layers:
            layer.trainable = False

        optimizer = keras.optimizers.Adam()
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        # 保存模型
        save_checkpoint = keras.callbacks.ModelCheckpoint(filepath='model.h5', monitor='val_loss', save_best_only=True,
                                                          verbose=1)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=True)

        history = model.fit(
            train_generator,
            steps_per_epoch=nb_train_samples // 16,
            epochs=5,
            callbacks=[save_checkpoint, early_stopping],
            validation_data=validation_generator,
            verbose=True,
            validation_steps=nb_validation_samples // 16
        )


if __name__ == "__main__":
    ds = DataSource()
    # ds.cnnModels()
    # ds.training()
