import random as rn

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn

class MaskedConv2D(tf.keras.layers.Conv2D):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2D, self).__init__(*args, **kwargs)

        assert mask_type in {'A', 'B'}
        self.mask_type = mask_type

    def build(self, input_shape):
        super(MaskedConv2D, self).build(input_shape)

        kH, kW = self.kernel_size
        mask = np.ones(self.kernel.shape, dtype=np.float32)
        mask[kH // 2, kW // 2 + (self.mask_type == 'B'):, :, :] = 0.
        mask[kH // 2 + 1:, :, :] = 0.


    def call(self, inputs):
        outputs = self._convolution_op(inputs, self.kernel)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


def main():
    random_seed = 42
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)
    rn.seed(random_seed)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    batch_size = 10
    train_buf = 60000

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=train_buf)
    train_dataset = train_dataset.batch(batch_size)

    inputs = keras.layers.Input(shape=(28, 28, 1))
    x = MaskedConv2D(mask_type='A', filters=16, kernel_size=3, strides=2, padding='same', activation='linear')(inputs)
    x = keras.layers.Activation(activation='relu')(x)
    x = MaskedConv2D(mask_type='B', filters=32, kernel_size=3, strides=2, padding='same', activation='linear')(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = MaskedConv2D(mask_type='B', filters=64, kernel_size=3, strides=2, padding='same', activation='linear')(x)
    x = keras.layers.Activation(activation='relu')(x)
    encoder = tf.keras.Model(inputs=inputs, outputs=x)


