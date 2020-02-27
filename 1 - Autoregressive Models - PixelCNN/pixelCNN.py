"""Script to train pixelCNN on the MNIST dataset."""
import random as rn
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import nn
from tensorflow.keras import initializers
from tensorflow.keras.utils import Progbar


class MaskedConv2D(tf.keras.layers.Layer):
    """Convolutional layers with masks.

    Convolutional layers with simple implementation of masks type A and B for
    autoregressive models.

    Arguments:
    mask_type: one of `"A"` or `"B".`
    filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
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

        assert mask_type in {'A', 'B'}
        self.mask_type = mask_type

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding.upper()
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        self.kernel = self.add_weight('kernel',
                                      shape=(self.kernel_size,
                                             self.kernel_size,
                                             int(input_shape[-1]),
                                             self.filters),
                                      initializer=self.kernel_initializer,
                                      trainable=True)

        self.bias = self.add_weight('bias',
                                    shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    trainable=True)

        center = self.kernel_size // 2

        mask = np.ones(self.kernel.shape, dtype=np.float32)
        mask[center, center + (self.mask_type == 'B'):, :, :] = 0.
        mask[center + 1:, :, :, :] = 0.

        self.mask = tf.constant(mask, dtype=tf.float32, name='mask')

    def call(self, input):
        masked_kernel = tf.math.multiply(self.mask, self.kernel)
        x = nn.conv2d(input,
                      masked_kernel,
                      strides=[1, self.strides, self.strides, 1],
                      padding=self.padding)
        x = nn.bias_add(x, self.bias)
        return x


class ResidualBlock(tf.keras.Model):
    """Residual blocks that compose pixelCNN

    Blocks of layers with 3 convolutional layers and one residual connection.
    Based on Figure 5 from [1] where h indicates number of filters.

    Refs:
    [1] - Oord, A. V. D., Kalchbrenner, N., & Kavukcuoglu, K. (2016). Pixel
     recurrent neural networks. arXiv preprint arXiv:1601.06759.
    """

    def __init__(self, h):
        super(ResidualBlock, self).__init__(name='')

        self.conv2a = keras.layers.Conv2D(filters=h, kernel_size=1, strides=1)
        self.conv2b = MaskedConv2D(mask_type='B', filters=h, kernel_size=3, strides=1)
        self.conv2c = keras.layers.Conv2D(filters=2 * h, kernel_size=1, strides=1)

    def call(self, input_tensor):
        x = tf.nn.relu(input_tensor)
        x = self.conv2a(x)

        x = tf.nn.relu(x)
        x = self.conv2b(x)

        x = tf.nn.relu(x)
        x = self.conv2c(x)

        x += input_tensor
        return x


def quantise(images, q_levels):
    """Quantise image into q levels."""
    return (np.digitize(images, np.arange(q_levels) / q_levels) - 1).astype('float32')


def main():
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

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train.reshape(x_train.shape[0], height, width, n_channel)
    x_test = x_test.reshape(x_test.shape[0], height, width, n_channel)

    # --------------------------------------------------------------------------------------------------------------
    # Quantise the input data in q levels
    q_levels = 2
    x_train_quantised = quantise(x_train, q_levels)
    x_test_quantised = quantise(x_test, q_levels)

    # --------------------------------------------------------------------------------------------------------------
    # Creating input stream using tf.data API
    batch_size = 128
    train_buf = 60000

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_quantised / (q_levels - 1),
                                                        x_train_quantised.astype('int32')))
    train_dataset = train_dataset.shuffle(buffer_size=train_buf)
    train_dataset = train_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_quantised / (q_levels - 1),
                                                       x_test_quantised.astype('int32')))
    test_dataset = test_dataset.batch(batch_size)

    # --------------------------------------------------------------------------------------------------------------
    # Create PixelCNN model
    inputs = keras.layers.Input(shape=(height, width, n_channel))
    x = MaskedConv2D(mask_type='A', filters=128, kernel_size=7, strides=1)(inputs)

    for i in range(15):
        x = ResidualBlock(h=64)(x)

    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1)(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1)(x)
    x = keras.layers.Conv2D(filters=q_levels, kernel_size=1, strides=1)(x)

    pixelcnn = tf.keras.Model(inputs=inputs, outputs=x)
    # --------------------------------------------------------------------------------------------------------------
    # Prepare optimizer and loss function
    lr_decay = 0.99995
    learning_rate = 1e-2
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    compute_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    # --------------------------------------------------------------------------------------------------------------
    @tf.function
    def train_step(batch_x, batch_y):
        with tf.GradientTape() as ae_tape:
            logits = pixelcnn(batch_x, training=True)

            loss = compute_loss(tf.one_hot(batch_y, q_levels), logits)

        gradients = ae_tape.gradient(loss, pixelcnn.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(zip(gradients, pixelcnn.trainable_variables))

        return loss

    # --------------------------------------------------------------------------------------------------------------
    # Training loop
    n_epochs = 150
    n_iter = int(np.ceil(x_train_quantised.shape[0] / batch_size))
    for epoch in range(n_epochs):
        progbar = Progbar(n_iter)
        print('Epoch {:}/{:}'.format(epoch + 1, n_epochs))

        for i_iter, (batch_x, batch_y) in enumerate(train_dataset):
            optimizer.lr = optimizer.lr * lr_decay
            loss = train_step(batch_x, batch_y)

            progbar.add(1, values=[('loss', loss)])
    # --------------------------------------------------------------------------------------------------------------
    # Test
    test_loss = []
    for batch_x, batch_y in test_dataset:
        logits = pixelcnn(batch_x, training=False)

        # Calculate cross-entropy (= negative log-likelihood)
        loss = compute_loss(tf.one_hot(batch_y, q_levels), logits)

        test_loss.append(loss)
    print('nll : {:} nats'.format(np.array(test_loss).mean()))
    print('bits/dim : {:}'.format(np.array(test_loss).mean() / (height * width)))

    # --------------------------------------------------------------------------------------------------------------
    # Generating new images
    samples = np.zeros((100, height, width, n_channel), dtype='float32')
    for i in range(height):
        for j in range(width):
            logits = pixelcnn(samples)
            next_sample = tf.random.categorical(logits[:, i, j, :], 1)
            samples[:, i, j, 0] = (next_sample.numpy() / (q_levels - 1))[:, 0]

    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        ax = fig.add_subplot(10, 10, i + 1)
        ax.matshow(samples[i, :, :, 0], cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()

    # --------------------------------------------------------------------------------------------------------------
    # Filling occluded images
    occlude_start_row = 14
    num_generated_images = 10
    samples = np.copy(x_test_quantised[0:num_generated_images, :, :, :])
    samples = samples / (q_levels - 1)
    samples[:, occlude_start_row:, :, :] = 0

    fig = plt.figure(figsize=(10, 10))

    for i in range(10):
        ax = fig.add_subplot(1, 10, i + 1)
        ax.matshow(samples[i, :, :, 0], cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))

    for i in range(occlude_start_row, height):
        for j in range(width):
            logits = pixelcnn(samples)
            next_sample = tf.random.categorical(logits[:, i, j, :], 1)
            samples[:, i, j, 0] = (next_sample.numpy() / (q_levels - 1))[:, 0]

    fig = plt.figure(figsize=(10, 10))

    for i in range(10):
        ax = fig.add_subplot(1, 10, i + 1)
        ax.matshow(samples[i, :, :, 0], cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()


if __name__ == '__main__':
    main()
