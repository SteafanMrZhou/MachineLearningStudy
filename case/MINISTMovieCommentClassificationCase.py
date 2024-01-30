# @Author   : Steafan
# @Desc     : 电影评论正面和负面分类问题
# @File     : MINISTMovieCommentClassificationCase.py
# @Create   : 2024-01-30 12:59

import tensorflow as tf
from tensorflow import keras
import numpy as np
import tensorflow_hub as hub
import tensorflow_datasets as tfds


train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])
(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews",
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True
)
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)