"""Implementation of the pixelcnn++ in the MNIST dataset."""

import random as rn
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# --------------------------------------------------------------------------------------------------------------
# Defining random seeds
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
rn.seed(random_seed)

# --------------------------------------------------------------------------------------------------------------
# Loading data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

height = 28
width = 28
n_channel = 1

x_train = (x_train.astype('float32') / 127.5) - 1
x_test = (x_test.astype('float32') / 127.5) - 1

x_train = x_train.reshape(x_train.shape[0], height, width, 1)
x_test = x_test.reshape(x_test.shape[0], height, width, 1)

x_train = x_train[y_train == 1]

batch_size = 64
train_buf = 10000

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train))
train_dataset = train_dataset.shuffle(buffer_size=train_buf)
train_dataset = train_dataset.batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, x_test))
test_dataset = test_dataset.batch(batch_size)


# --------------------------------------------------------------------------------------------------------------
class DownShiftedConv2d(tf.keras.Model):
    """"""

    def __init__(self, filters, kernel_size, stride=[1,1]):
        super(DownShiftedConv2d, self).__init__(name='')
        self.vertical_padding = keras.layers.ZeroPadding2D(padding=((kernel_size // 2 + 1, 0),
                                                                    (kernel_size // 2, kernel_size // 2)))

        self.vertical_conv = keras.layers.Conv2D(filters=filters,
                                                 kernel_size=[kernel_size // 2 + 1, kernel_size],
                                                 strides=1,
                                                 padding='valid', kernel_initializer='ones')

        self.vertical_cropping = keras.layers.Cropping2D(cropping=((0, 1), (0, 0)))

    def call(self, input_tensor):
        vertical_preactivation = self.vertical_padding(input_tensor)
        vertical_preactivation = self.vertical_conv(vertical_preactivation)
        output = self.vertical_cropping(vertical_preactivation)

        return output


class DownRightShiftedConv2d(tf.keras.Model):
    """"""

    def __init__(self, mask_type, filters, kernel_size, stride=[1,1]):
        super(DownRightShiftedConv2d, self).__init__(name='')
        self.mask_type = mask_type

        self.horizontal_padding = keras.layers.ZeroPadding2D(padding=((0, 0), (kernel_size // 2 + 1, 0)))
        self.horizontal_conv = keras.layers.Conv2D(filters=filters,
                                                   kernel_size=[1, kernel_size // 2 + 1],
                                                   strides=1,
                                                   padding='valid', kernel_initializer='ones')
        if mask_type == 'B':
            self.horizontal_cropping = keras.layers.Cropping2D(cropping=((0, 0), (1, 0)))
        elif mask_type == 'A':
            self.horizontal_cropping = keras.layers.Cropping2D(cropping=((0, 0), (0, 1)))

    def call(self, input_tensor):
        horizontal_preactivation = self.horizontal_padding(input_tensor)
        horizontal_preactivation = self.horizontal_conv(horizontal_preactivation)
        output = self.horizontal_cropping(horizontal_preactivation)

        return output

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape())-1
    return tf.nn.elu(tf.concat([x, -x], axis))


nr_logistic_mix = 5
filters = 160
kernel_size = 3
nr_resnet = 6
dropout_p = 0.5

inputs = keras.layers.Input(shape=(height, width, n_channel))
u = DownShiftedConv2d(filters, kernel_size)(inputs)
ul = DownRightShiftedConv2d('A', filters, kernel_size)(inputs)

for rep in range(nr_resnet):
    #  GatedResnet
    uc1 = DownShiftedConv2d(filters, kernel_size)(concat_elu(u))
    uc1 = concat_elu(uc1)
    uc1 = tf.nn.dropout(uc1, keep_prob=1. - dropout_p)
    uc2 = DownShiftedConv2d(filters, kernel_size)(uc1)
    a, b = tf.split(uc2, 2, 3)
    uc3 = a * tf.nn.sigmoid(b)
    u = u + uc3

    #  GatedResnet
    ulc1 = DownRightShiftedConv2d('B', filters, kernel_size)(concat_elu(u))
    ulc1 = concat_elu(ulc1)
    ulc1 = tf.nn.dropout(ulc1, keep_prob=1. - dropout_p)
    ulc2 = DownRightShiftedConv2d('B', filters, kernel_size)(ulc1)
    a, b = tf.split(ulc2, 2, 3)
    ulc3 = a * tf.nn.sigmoid(b)
    ul = ul + ulc3

    DownShiftedConv2d(filters, kernel_size, stride=[2, 2])
    DownRightShiftedConv2d('B', filters, kernel_size, stride=[2, 2])

    u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
    ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))