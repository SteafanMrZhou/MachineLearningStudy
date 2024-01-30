# @Author   : Steafan
# @Desc     : 
# @File     : FashionMINISTClassifyImageCase.py
# @Create   : 2024-01-30 11:07

import tensorflow as tf
from tensorflow import keras

import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# print(len(train_labels))
# print(len(train_images.shape))
# print(train_images.shape)

train_images = train_images / 255.0
test_images = test_images / 255.0

# 构建模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 训练模型
model.fit(train_images, train_labels, epochs=10)
# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\n Test accuracy:', test_acc)
# 预测模型
predictions = model.predict(test_images)
print(np.argmax(predictions[0]))
print(test_labels[0])