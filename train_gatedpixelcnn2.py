"""
https://github.com/jakebelew/gated-pixel-cnn/blob/master/ops.py
https://github.com/chandlersupple/Tensorflow-PixelCNN/blob/master/pixelcnn.py
"""
import random as rn

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class MaskedConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 mask_type,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        super(MaskedConv2D, self).__init__()

        assert mask_type in {'A', 'B'}
        self.mask_type = mask_type

        self.strides = strides
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding.upper()
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        self.kernel = self.add_variable("kernel",
                                        shape=(self.kernel_size,
                                               self.kernel_size,
                                               int(input_shape[-1]),
                                               self.filters),
                                        initializer=self.kernel_initializer,
                                        trainable=True)

        self.bias = self.add_variable("bias",
                                      shape=(self.filters, ),
                                      initializer=self.bias_initializer,
                                      trainable=True)

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


class ResidualBlock(tf.keras.Model):
    def __init__(self, h):
        super(ResidualBlock, self).__init__(name='')

        self.conv2a = MaskedConv2D(mask_type='B', filters=h, kernel_size=1, strides=1)
        self.conv2b = MaskedConv2D(mask_type='B', filters=h, kernel_size=3, strides=1)
        self.conv2c = MaskedConv2D(mask_type='B', filters=2*h, kernel_size=1, strides=1)

    def call(self, input_tensor):
        x = tf.nn.relu(input_tensor)
        x = self.conv2a(x)

        x = tf.nn.relu(x)
        x = self.conv2b(x)

        x = tf.nn.relu(x)
        x = self.conv2c(x)

        x += input_tensor
        return x

levels = 256
def quantisize(images, levels):
    return (np.digitize(images, np.arange(levels) / levels) - 1).astype('i')


def binarize(images):
    """
    Stochastically binarize values in [0, 1] by treating them as p-values of
    a Bernoulli distribution.
    """
    return (np.random.uniform(size=images.shape) < images).astype('float32')


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

    levels = 2
    x_train_quantised = quantisize(x_train,levels)
    x_test_quantised = quantisize(x_test,levels)

    # discretization_step = 1. / (levels - 1)
    # x_train_quantised = x_train_quantised * discretization_step
    # x_test_quantised = x_test_quantised * discretization_step


    batch_size = 100
    train_buf = 60000

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train_quantised))
    train_dataset = train_dataset.shuffle(buffer_size=train_buf)
    train_dataset = train_dataset.batch(batch_size)

    # https://github.com/RishabGoel/PixelCNN/blob/master/pixel_cnn.py
    # https://github.com/jonathanventura/pixelcnn/blob/master/pixelcnn.py
    n_channel = 1
    q_levels = 2

    inputs = keras.layers.Input(shape=(28, 28, 1))
    x = MaskedConv2D(mask_type='A', filters=128, kernel_size=7, strides=1)(inputs)
    for i in range(15):
        x = ResidualBlock(h=64)(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = MaskedConv2D(mask_type='B', filters=256, kernel_size=1, strides=1)(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = MaskedConv2D(mask_type='B', filters=n_channel * q_levels, kernel_size=1, strides=1)(x)  # shape [N,H,W,DC]

    pixelcnn = tf.keras.Model(inputs=inputs, outputs=x)

    learning_rate = 3e-4
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_step(batch_x, batch_y):
        with tf.GradientTape() as ae_tape:
            logits = pixelcnn(batch_x)

            logits = tf.reshape(logits, [-1, 28, 28, q_levels, n_channel])  # shape [N,H,W,DC] -> [N,H,W,D,C]
            logits = tf.transpose(logits, perm=[0, 1, 2, 4, 3])  # shape [N,H,W,D,C] -> [N,H,W,C,D]

            flattened_logits = tf.reshape(logits, [-1, q_levels])  # [N,H,W,C,D] -> [NHWC,D]
            target_pixels_loss = tf.reshape(batch_y, [-1,1])  # [N,H,W,C] -> [NHWC]

            loss  = compute_loss(target_pixels_loss, flattened_logits)
        ae_grads = ae_tape.gradient(loss, pixelcnn.trainable_variables)
        # ae_grads, _ = tf.clip_by_norm(ae_grads, 1)
        optimizer.apply_gradients(zip(ae_grads, pixelcnn.trainable_variables))

        return loss

    epochs = 10
    for epoch in range(epochs):
        print(epoch)
        for batch_x, batch_y in train_dataset:
            # print()
            loss = train_step(batch_x, batch_y)
            print(loss)




    samples = np.zeros((1, 28, 28, 1), dtype='float32')
    for i in range(28):
        for j in range(28):
            print("{} {}".format(i, j))
            A = pixelcnn(samples)
            next_sample = binarize(A.numpy())
            print(next_sample[:, i, j, 0])
            samples[:, i, j, 0] = next_sample[:, i, j, 0]

