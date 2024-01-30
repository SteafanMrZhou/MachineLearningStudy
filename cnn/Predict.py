# @Author   : Steafan
# @Desc     : 
# @File     : Predict.py
# @Create   : 2024-01-30 16:05

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image

from cnn.CNN import CNN

class Predict(object):
    def __init__(self):
        latest = tf.train.latest_checkpoint('./ckpt')
        self.cnn = CNN()
        self.cnn.model.load_weights(latest)

    def predict(self, image_path):
        img = Image.open(image_path).convert('L')
        img = np.reshape(img, (28, 28, 1)) / 255.0
        x = np.array([1 - img])

        y = self.cnn.model.predict(x)
        print(image_path)
        print(y[0])
        print('        -> Predict digit', np.argmax(y[0]))



if __name__ == '__main__':
    app = Predict()
    app.predict('./test_images/0.png')
    app.predict('./test_images/1.png')
    app.predict('./test_images/4.png')