"""Script to compare masked vs cropped conv2d"""
import random as rn
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
rn.seed(random_seed)

# ---------------------------------------------------------------------------------
class MaskedConv2D(tf.keras.layers.Layer):
    """Convolutional layers with masks for autoregressive models

    Convolutional layers with simple implementation to have masks type A and B.
    """

    def __init__(self,
                 mask_type,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        super(MaskedConv2D, self).__init__()

        assert mask_type in {'A', 'B', 'V'}
        self.mask_type = mask_type

        self.filters = filters

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        self.strides = strides
        self.padding = padding.upper()
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        kernel_h, kernel_w = self.kernel_size

        self.kernel = self.add_weight("kernel",
                                      shape=(kernel_h,
                                             kernel_w,
                                             int(input_shape[-1]),
                                             self.filters),
                                      initializer=self.kernel_initializer,
                                      trainable=True)

        self.bias = self.add_weight("bias",
                                    shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    trainable=True)

        mask = np.ones(self.kernel.shape, dtype=np.float32)
        if self.mask_type == 'V':
            mask[kernel_h // 2:, :, :, :] = 0.
        else:
            mask[kernel_h // 2, kernel_w // 2 + (self.mask_type == 'B'):, :, :] = 0.
            mask[kernel_h // 2 + 1:, :, :] = 0.

        self.mask = tf.constant(mask, dtype=tf.float32, name='mask')

    def call(self, input):
        masked_kernel = tf.math.multiply(self.mask, self.kernel)
        x = tf.nn.conv2d(input, masked_kernel, strides=[1, self.strides, self.strides, 1], padding=self.padding)
        x = tf.nn.bias_add(x, self.bias)
        return x

class MaskedGatedBlock(tf.keras.Model):
    """"""
    def __init__(self, mask_type, filters, kernel_size):
        super(MaskedGatedBlock, self).__init__(name='')

        self.mask_type = mask_type
        self.vertical_conv = MaskedConv2D(mask_type='V', filters=2 * filters, kernel_size=kernel_size, kernel_initializer='ones')
        self.horizontal_conv = MaskedConv2D(mask_type=mask_type, filters=2 * filters, kernel_size=(1, kernel_size), kernel_initializer='ones')
        self.v_to_h_conv = keras.layers.Conv2D(filters=2 * filters, kernel_size=1, kernel_initializer='ones')

        self.horizontal_output = keras.layers.Conv2D(filters=filters, kernel_size=1, kernel_initializer='ones')

    def _gate(self, x):
        tanh_preactivation, sigmoid_preactivation = tf.split(x, 2, axis=-1)
        return tf.nn.tanh(tanh_preactivation) * tf.nn.sigmoid(sigmoid_preactivation)

    def call(self, input_tensor):
        v = input_tensor[0]
        h = input_tensor[1]

        horizontal_preactivation = self.horizontal_conv(h)  # 1xN
        vertical_preactivation = self.vertical_conv(v)  # NxN
        v_to_h = self.v_to_h_conv(vertical_preactivation)  # 1x1
        v_out = self._gate(vertical_preactivation)

        horizontal_preactivation = horizontal_preactivation + v_to_h
        h_activated = self._gate(horizontal_preactivation)
        h_activated = self.horizontal_output(h_activated)

        if self.mask_type == 'A':
            h_out = h_activated
        elif self.mask_type == 'B':
            h_out = h + h_activated

        return v_out, h_out

# ---------------------------------------------------------------------------------
class DownShiftedConv2d(tf.keras.Model):
    """"""
    def __init__(self, filters, kernel_size):
        super(DownShiftedConv2d, self).__init__(name='')
        self.vertical_padding = keras.layers.ZeroPadding2D(padding=((kernel_size//2+1, 0),
                                                                    (kernel_size//2, kernel_size//2)))

        self.vertical_conv = keras.layers.Conv2D(filters=filters,
                                                 kernel_size=[kernel_size//2+1, kernel_size],
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
    def __init__(self, mask_type, filters, kernel_size):
        super(DownRightShiftedConv2d, self).__init__(name='')
        self.mask_type = mask_type

        self.horizontal_padding = keras.layers.ZeroPadding2D(padding=((0, 0), (kernel_size // 2+1, 0)))
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



class CroppedGatedBlock(tf.keras.Model):
    """"""
    def __init__(self, mask_type, filters, kernel_size):
        super(CroppedGatedBlock, self).__init__(name='')

        self.mask_type = mask_type
        self.vertical_stream = DownShiftedConv2d(2*filters, kernel_size)
        self.horizontal_stream = DownRightShiftedConv2d(mask_type, 2*filters, kernel_size)

        self.v_to_h_conv = keras.layers.Conv2D(filters=2 * filters, kernel_size=1, kernel_initializer='ones')

        self.horizontal_output = keras.layers.Conv2D(filters=filters, kernel_size=1, kernel_initializer='ones')


    def _gate(self, x):
        tanh_preactivation, sigmoid_preactivation = tf.split(x, 2, axis=-1)
        return tf.nn.tanh(tanh_preactivation) * tf.nn.sigmoid(sigmoid_preactivation)

    def call(self, input_tensor):
        v = input_tensor[0]
        h = input_tensor[1]

        vertical_preactivation = self.vertical_stream(v)
        horizontal_preactivation = self.horizontal_stream(h)
        v_to_h = self.v_to_h_conv(vertical_preactivation)  # 1x1
        v_out = self._gate(vertical_preactivation)

        horizontal_preactivation = horizontal_preactivation + v_to_h
        h_activated = self._gate(horizontal_preactivation)
        h_activated = self.horizontal_output(h_activated)

        if self.mask_type == 'A':
            h_out = h_activated
        elif self.mask_type == 'B':
            h_out = h + h_activated

        return v_out, h_out

# ---------------------------------------------------------------------------------
CroppedGatedBlock('B', 1, 3)
MaskedGatedBlock('B', 1, 3)