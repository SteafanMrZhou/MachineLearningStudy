# @Author   : Steafan
# @Desc     : 猫狗识别数据源处理+模型实现
# @File     : DataSource.py
# @Create   : 2024-02-01 15:03

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from sklearn.metrics import  confusion_matrix
import numpy as np
import pathlib as path
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)


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
                           nb_test_samples=nb_test_samples,
                           test_generator=test_generator,
                           classes=classes)

    def cnnModels(self, img_width, img_height, num_classes, train_generator, validation_generator, nb_train_samples,
                  nb_validation_samples, nb_test_samples, test_generator, classes):
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
        # 训练结果和损失、准确率可视化展示
        self.trainingAndValidationLossAndAccuracyVisualization(history)
        # 模型评估
        validation_evaluate_score = model.evaluate(validation_generator, verbose=False)
        print('Val loss:', validation_evaluate_score[0])
        print('Val accuracy:', validation_evaluate_score[1])
        test_evaluate_score = model.evaluate(test_generator, verbose=False)
        print('Test loss:', test_evaluate_score[0])
        print('Test accuracy:', test_evaluate_score[1])
        # 混淆矩阵
        y_pred = np.argmax(model.predict(test_generator), axis=1)
        cm = confusion_matrix(test_generator.classes, y_pred)
        # 热力图
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cbar=True, cmap='Blues', xticklabes=classes, yticklabels=classes)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion Matrix')
        plt.show()

    # 将模型训练和验证的损失可视化出来、以及训练和验证的准确率
    def trainingAndValidationLossAndAccuracyVisualization(self, history):
        history_dic = history.history
        loss_values = history_dic['loss']
        val_loss_values = history_dic['val_loss']
        epochs_x = range(1, len(loss_values) + 1)
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 1, 1)
        plt.plot(epochs_x, loss_values, 'b-o', label='Training loss')
        plt.plot(epochs_x, val_loss_values, 'r-o', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.subplot(2, 1, 2)
        acc_values = history_dic['accuracy']
        val_acc_values = history_dic['val_accuracy']
        plt.plot(epochs_x, acc_values, 'b-o', label='Training acc')
        plt.plot(epochs_x, val_acc_values, 'r-o', label='Validation acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Acc')
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    ds = DataSource()
    # ds.cnnModels()
    # ds.training()
