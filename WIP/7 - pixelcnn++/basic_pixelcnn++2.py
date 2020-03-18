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


class DownShiftedConv2d(tf.keras.Model):
    """"""

    def __init__(self,
                 num_filters,
                 filter_size=[2, 3],
                 stride=[1, 1]):
        super(DownShiftedConv2d, self).__init__(name='')

        self.conv = keras.layers.Conv2D(num_filters, kernel_size=filter_size, padding='valid', strides=stride)
        self.filter_size = filter_size

    def call(self, input_tensor):

        x = tf.pad(input_tensor,
                   [[0, 0],
                    [self.filter_size[0] - 1, 0],
                    [int((self.filter_size[1] - 1) / 2), int((self.filter_size[1] - 1) / 2)],
                    [0, 0]])
        x = self.conv(x)
        return x



class DownRightShiftedConv2d(tf.keras.Model):
    """"""

    def __init__(self,
                 num_filters,
                 filter_size=[2, 2],
                 stride=[1, 1]):
        super(DownShiftedConv2d, self).__init__(name='')

        self.conv = keras.layers.Conv2D(num_filters, kernel_size=filter_size, padding='valid', strides=stride)
        self.filter_size = filter_size

    def call(self, input_tensor):
        x = tf.pad(input_tensor,
                   [[0, 0],
                    [self.filter_size[0] - 1, 0],
                    [self.filter_size[1] - 1, 0],
                    [0, 0]])
        x = self.conv(x)
        return x


def nin(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """



class GatedResnet(tf.keras.Model):
    """"""

    def __init__(self,
                 conv_type,
                 num_filters,
                 filter_size,
                 stride,
                 dropout_p=0.):
        super(GatedResnet, self).__init__(name='')

        if conv_type == 'd':
            self.conv1 = keras.layers.Conv2D(num_filters, kernel_size=filter_size, padding='valid', strides=stride)
            self.conv2 = keras.layers.Conv2D(num_filters, kernel_size=filter_size, padding='valid', strides=stride)

        elif conv_type == 'd':
            self.conv1 = keras.layers.Conv2D(num_filters, kernel_size=filter_size, padding='valid', strides=stride)
            self.conv2 = keras.layers.Conv2D(num_filters, kernel_size=filter_size, padding='valid', strides=stride)

        elif conv_type == 'd':
            self.conv1 = keras.layers.Conv2D(num_filters, kernel_size=filter_size, padding='valid', strides=stride)
            self.conv2 = keras.layers.Conv2D(num_filters, kernel_size=filter_size, padding='valid', strides=stride)

        else:
            self.conv1 = keras.layers.Conv2D(num_filters, kernel_size=filter_size, padding='valid', strides=stride)
            self.conv2 = keras.layers.Conv2D(num_filters, kernel_size=filter_size, padding='valid', strides=stride)

        self.filter_size = filter_size

    def concat_elu(x):
        """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
        axis = len(x.get_shape()) - 1
        return tf.nn.elu(tf.concat([x, -x], axis))

    def call(self, input_tensor):
        x, a = input_tensor
        c1 = self.conv1(self.concat_elu(input_tensor))

        if a is not None:  # add short-cut connection if auxiliary input 'a' is given
            # s = x.shape
            # x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
            # x = keras.layers.Dense(num_units, **kwargs)(x)
            # c1 += tf.reshape(x, s[:-1] + [num_units])

            c1 += nin(self.concat_elu(a))

        c1 = self.concat_elu(c1)

        if self.dropout_p > 0:
            c1 = tf.nn.dropout(c1, keep_prob=1. - self.dropout_p)

        c2 = self.conv2(c1)

        a, b = tf.split(c2, 2, 3)
        c3 = a * tf.nn.sigmoid(b)
        return x + c3




class PixelPP(tf.keras.Model):
    """"""

    def __init__(self,
                 nr_resnet):
        super(UpPassModel, self).__init__(name='')

        self.nr_resnet = nr_resnet
        self.u_list = []
        self.ul_list = []

        for rep in range(nr_resnet):
            self.u_list.append(GatedResnet())
            self.ul_list.append(GatedResnet())
        self.u_list.append(GatedResnet())
        self.ul_list.append(GatedResnet())
        for rep in range(nr_resnet):
            self.u_list.append(GatedResnet())
            self.ul_list.append(GatedResnet())
        self.u_list.append(GatedResnet())
        self.ul_list.append(GatedResnet())
        for rep in range(nr_resnet):
            self.u_list.append(GatedResnet())
            self.ul_list.append(GatedResnet())

        self.u_list = []
        self.ul_list = []

        for rep in range(nr_resnet):
            self.u_list.append(GatedResnet())
            self.ul_list.append(GatedResnet())
        self.u_list.append(GatedResnet())
        self.ul_list.append(GatedResnet())
        for rep in range(nr_resnet+1):
            self.u_list.append(GatedResnet())
            self.ul_list.append(GatedResnet())
        self.u_list.append(GatedResnet())
        self.ul_list.append(GatedResnet())
        for rep in range(nr_resnet+1):
            self.u_list.append(GatedResnet())
            self.ul_list.append(GatedResnet())


    def up_pass(self):
        # ////////// up pass through pixelCNN ////////
        xs = nn.int_shape(x)
        x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])],
                          3)  # add channel of ones to distinguish image from padding later on
        u_list = [nn.down_shift(
            nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))]  # stream for pixels above
        ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) + \
                   nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters,
                                                               filter_size=[2, 1]))]  # stream for up and to the left

        for rep in range(nr_resnet):
            u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
            ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

        u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
        ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))

        for rep in range(nr_resnet):
            u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
            ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

        u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
        ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))

        for rep in range(nr_resnet):
            u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
            ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

    def down_pass(self):
        # /////// down pass ////////
        u = u_list.pop()
        ul = ul_list.pop()
        for rep in range(nr_resnet):
            u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
            ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)

        u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
        ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])

        for rep in range(nr_resnet + 1):
            u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
            ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)

        u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
        ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])

        for rep in range(nr_resnet + 1):
            u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
            ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()], 3), conv=nn.down_right_shifted_conv2d)


    def call(self, input_tensor):
        up_pass()
        down_pass()
        pass
