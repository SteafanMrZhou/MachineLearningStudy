# @Author   : Steafan
# @Desc     : 
# @File     : Train.py
# @Create   : 2024-01-30 15:55

import tensorflow as tf
from tensorflow import keras

from cnn.CNN import CNN
from cnn.DataSource import DataSource

class Train:
    def __init__(self):
        self.cnn = CNN()
        self.data = DataSource()

    def train(self):
        check_path = './ckpt/cp-{epoch:04d}.ckpt'
        save_model_cb = keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True, verbose=1, period=5)
        self.cnn.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics='accuracy'
                               )
        self.cnn.model.fit(self.data.train_images, self.data.train_labels, epochs=5, callbacks=[save_model_cb])

        test_loss, test_acc = self.cnn.model.evaluate(self.data.test_images, self.data.test_labels)
        print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(self.data.test_labels)))



if __name__ == '__main__':
    app = Train()
    app.train()