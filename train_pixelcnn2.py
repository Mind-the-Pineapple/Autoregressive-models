import random as rn

import tensorflow as tf
from tensorflow import keras
import numpy as np


class MaskedConv2D(tf.keras.layers.Layer):
    def __init__(self, mask_type, filters, kernel_size, strides=1, padding='same'):
        super(MaskedConv2D, self).__init__()

        assert mask_type in {'A', 'B'}
        self.mask_type = mask_type

        self.strides = strides
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding.upper()

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=[self.kernel_size,
                                               self.kernel_size,
                                               int(input_shape[-1]),
                                               self.filters])

        self.bias = self.add_variable("bias",
                                      shape=[self.filters, ])

        mask = np.ones(self.kernel.shape, dtype=np.float32)
        mask[self.kernel_size // 2, self.kernel_size // 2 + (self.mask_type == 'B'):, :, :] = 0.
        mask[self.kernel_size // 2 + 1:, :, :] = 0.

        self.mask = tf.constant(mask,
                                dtype=tf.float32,
                                name='mask')

    def call(self, input):
        masked_kernel = tf.math.multiply(self.mask, self.kernel)
        x = tf.nn.conv2d(input, masked_kernel, strides=[1, self.strides, self.strides, 1], padding=self.padding)
        x = tf.nn.bias_add(x, self.bias)
        return x


def binarize(x):
    x[x >= .5] = 1.
    x[x < .5] = 0.
    return x


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

    # Binarization
    x_train = binarize(x_train)
    x_test = binarize(x_test)

    batch_size = 10
    train_buf = 60000

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=train_buf)
    train_dataset = train_dataset.batch(batch_size)

    inputs = keras.layers.Input(shape=(28, 28, 1))
    x = MaskedConv2D(mask_type='A', filters=16, kernel_size=7, strides=1)(inputs)
    x = MaskedConv2D(mask_type='B', filters=32, kernel_size=3, strides=1)(x)
    x = MaskedConv2D(mask_type='B', filters=1, kernel_size=3, strides=1)(x)
    encoder = tf.keras.Model(inputs=inputs, outputs=x)

    learning_rate = 3e-4
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    @tf.function
    def train_step(x):
        with tf.GradientTape(persistent=True) as ae_tape:
            z = encoder(x)

            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=x, name='loss'))
        ae_grads = ae_tape.gradient(loss, encoder.trainable_variables)
        optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables))

        return loss

    epochs = 50
    for epoch in range(epochs):
        print(epoch)
        for x in train_dataset:
            loss = train_step(x)
            # print(loss)

    with tf.device('/GPU:1'):

        samples = np.zeros((1, 28, 28, 1), dtype='float32')

        for i in range(28):
            for j in range(28):
                print("{} {}".format(i, j))
                A = encoder(samples)
                next_sample = binarize(A.numpy())
                print(next_sample[:, i, j, 0])
                samples[:, i, j, 0] = next_sample[:, i, j, 0]
