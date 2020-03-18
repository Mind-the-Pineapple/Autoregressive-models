""""""
import random as rn

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def main():
    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    rn.seed(random_seed)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.