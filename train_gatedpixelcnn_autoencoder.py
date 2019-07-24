import random as rn

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class MaskedConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 mask_type='B',
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        super(MaskedConv2D, self).__init__()

        assert mask_type in {'A', 'B', 'V'}
        self.mask_type = mask_type

        self.strides = strides
        self.filters = filters

        if isinstance(kernel_size, int):
            kernel_size =  (kernel_size,) * 2
        self.kernel_size = kernel_size

        self.padding = padding.upper()
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        kernel_h, kernel_w = self.kernel_size

        self.kernel = self.add_variable("kernel",
                                        shape=(kernel_h,
                                               kernel_w,
                                               int(input_shape[-1]),
                                               self.filters),
                                        initializer=self.kernel_initializer,
                                        trainable=True)

        self.bias = self.add_variable("bias",
                                      shape=(self.filters, ),
                                      initializer=self.bias_initializer,
                                      trainable=True)

        mask = np.ones(self.kernel.shape, dtype=np.float32)
        if self.mask_type == 'V':
            mask[kernel_h // 2:, :, :, :] = 0.
        else:
            mask[kernel_h // 2, kernel_w // 2 + (self.mask_type == 'B'):, :, :] = 0.
            mask[kernel_h // 2 + 1:, :, :] = 0.

        self.mask = tf.constant(mask,
                                dtype=tf.float32,
                                name='mask')

    def call(self, input):
        masked_kernel = tf.math.multiply(self.mask, self.kernel)
        x = tf.nn.conv2d(input, masked_kernel, strides=[1, self.strides, self.strides, 1], padding=self.padding)
        x = tf.nn.bias_add(x, self.bias)
        return x


def _gate(x):
    tanh_preactivation, sigmoid_preactivation =  tf.split(x, 2, axis=-1)
    return tf.nn.tanh(tanh_preactivation) * tf.nn.sigmoid(sigmoid_preactivation)

def gated_block_pass(n_filters, kernel_size, input_tensor, y):
    v, h = tf.split(input_tensor, 2, axis=-1)

    # TODO: 1×1convolution applied to a 1-hot encoding.
    codified = keras.layers.Conv2D(filters=2 * n_filters, kernel_size=1)(y)

    horizontal_preactivation = MaskedConv2D(2 * n_filters, kernel_size=(1, kernel_size))(h)  # 1xN
    vertical_preactivation = MaskedConv2D(2 * n_filters, kernel_size=(kernel_size, kernel_size), mask_type='V')(v)  # NxN
    v_to_h = keras.layers.Conv2D(filters=2 * n_filters, kernel_size=1)(vertical_preactivation)  # 1x1
    vertical_preactivation = vertical_preactivation + codified
    v_out = _gate(vertical_preactivation)
    horizontal_preactivation = horizontal_preactivation + v_to_h
    horizontal_preactivation = horizontal_preactivation + codified
    h_activated = _gate(horizontal_preactivation)
    h_preres = keras.layers.Conv2D(filters=n_filters, kernel_size=1)(h_activated)
    h_out = h + h_preres
    output = tf.concat((v_out, h_out), axis=-1)
    return output



def quantisize(images, levels):
    return (np.digitize(images, np.arange(levels) / levels) - 1).astype('int32')


def binarize(images):
    """
    Stochastically binarize values in [0, 1] by treating them as p-values of
    a Bernoulli distribution.
    """
    return (np.random.uniform(size=images.shape) < images).astype('float32')

def sample_from(distribution):
    batch_size, bins = distribution.shape
    return np.array([np.random.choice(bins, p=distr) for distr in distribution])

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

    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')


    q_levels = 2
    x_train_quantised = quantisize(x_train,q_levels)
    x_test_quantised = quantisize(x_test,q_levels)

    batch_size = 10
    train_buf = 60000

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_quantised.astype('float32')/(q_levels-1), x_train_quantised))
    train_dataset = train_dataset.shuffle(buffer_size=train_buf)
    train_dataset = train_dataset.batch(batch_size)

    # https://github.com/RishabGoel/PixelCNN/blob/master/pixel_cnn.py
    # https://github.com/jonathanventura/pixelcnn/blob/master/pixelcnn.py
    n_channel = 1

    inputs = keras.layers.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(filters=100, kernel_size=5, strides=1, activation='relu')(inputs)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2,padding='same')(x)
    x = keras.layers.Conv2D(filters=150, kernel_size=5, strides=1, activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2,padding='same')(x)
    x = keras.layers.Conv2D(filters=200, kernel_size=3, strides=1, activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2,padding='same')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10, activation='linear')(x)

    y = tf.broadcast_to(tf.expand_dims(tf.expand_dims(x, 1), 1), [batch_size, 28, 28, 10])

    x = MaskedConv2D(mask_type='A', filters=256, kernel_size=7, strides=1)(inputs)
    for i in range(7):
        x = gated_block_pass(n_filters=128, kernel_size=3, input_tensor=x, y=y)
    v, h = tf.split(x, 2, axis=-1)
    x = keras.layers.Activation(activation='relu')(h)
    x = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1)(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1)(x)
    x = keras.layers.Conv2D(filters=n_channel * q_levels, kernel_size=1, strides=1)(x)  # shape [N,H,W,DC]

    pixelcnn = tf.keras.Model(inputs=inputs, outputs=x)

    learning_rate = 3e-4
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    compute_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_step(bla):
        batch_x, batch_y = bla
        with tf.GradientTape() as ae_tape:
            logits = pixelcnn(batch_x)

            logits = tf.reshape(logits, [-1, 28, 28, q_levels, n_channel])  # shape [N,H,W,DC] -> [N,H,W,D,C]
            logits = tf.transpose(logits, perm=[0, 1, 2, 4, 3])  # shape [N,H,W,D,C] -> [N,H,W,C,D]

            loss  = compute_loss(tf.one_hot(batch_y, q_levels) , logits)

        gradients = ae_tape.gradient(loss, pixelcnn.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(zip(gradients, pixelcnn.trainable_variables))

        return loss

    epochs = 5
    for epoch in range(epochs):
        print(epoch)
        for batch_x in train_dataset:
            print()
            loss = train_step(batch_x)
            print(loss)

