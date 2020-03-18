"""
https://github.com/kkleidal/GatedPixelCNNPyTorch/blob/master/models/components/pixelcnn.py?fbclid=IwAR2CVKrEmuT7XygEhRJgKi2aXAM8cgL6ttDuguKIUKWeKt5sNTTf4JKBqH0
Pixel Recursive Super Resolution
Probabilistic Semantic Inpainting with Pixel Constrained CNNs
"""
import random as rn
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


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

        self.mask = tf.constant(mask,
                                dtype=tf.float32,
                                name='mask')

    def call(self, input):
        masked_kernel = tf.math.multiply(self.mask, self.kernel)
        x = tf.nn.conv2d(input, masked_kernel, strides=[1, self.strides, self.strides, 1], padding=self.padding)
        x = tf.nn.bias_add(x, self.bias)
        return x


class GatedBlock(tf.keras.Model):
    """"""

    def __init__(self, mask_type, filters, kernel_size):
        super(GatedBlock, self).__init__(name='')

        self.mask_type = mask_type
        self.vertical_conv = MaskedConv2D(mask_type='V', filters=2 * filters, kernel_size=kernel_size)
        self.horizontal_conv = MaskedConv2D(mask_type=mask_type, filters=2 * filters, kernel_size=(1, kernel_size))
        self.v_to_h_conv = keras.layers.Conv2D(filters=2 * filters, kernel_size=1)

        self.horizontal_output = keras.layers.Conv2D(filters=filters, kernel_size=1)

        self.cond_fc_h = keras.layers.Dense(2 * filters, use_bias=False)
        self.cond_fc_v = keras.layers.Dense(2 * filters, use_bias=False)

        self.cond_conv_h = keras.layers.Conv2D(filters=2 * filters, kernel_size=1, use_bias=False)
        self.cond_conv_v = keras.layers.Conv2D(filters=2 * filters, kernel_size=1, use_bias=False)

    def _gate(self, x):
        tanh_preactivation, sigmoid_preactivation = tf.split(x, 2, axis=-1)
        return tf.nn.tanh(tanh_preactivation) * tf.nn.sigmoid(sigmoid_preactivation)

    def call(self, input_tensor):
        v, h, conditional_vector, conditional_image = input_tensor

        conditional_vector = tf.one_hot(conditional_vector, 10)
        conditional_vector_h = tf.expand_dims(tf.expand_dims(self.cond_fc_h(conditional_vector), 1), 1)
        conditional_vector_v = tf.expand_dims(tf.expand_dims(self.cond_fc_v(conditional_vector), 1), 1)

        conditional_image_h = self.cond_conv_h(conditional_image)
        conditional_image_v = self.cond_conv_v(conditional_image)

        horizontal_preactivation = self.horizontal_conv(h)
        vertical_preactivation = self.vertical_conv(v)
        v_to_h = self.v_to_h_conv(vertical_preactivation)
        vertical_preactivation = vertical_preactivation + conditional_vector_v
        vertical_preactivation = vertical_preactivation + conditional_image_v

        v_out = self._gate(vertical_preactivation)

        horizontal_preactivation = horizontal_preactivation + v_to_h
        horizontal_preactivation = horizontal_preactivation + conditional_vector_h
        horizontal_preactivation = horizontal_preactivation + conditional_image_h
        h_activated = self._gate(horizontal_preactivation)

        if self.mask_type == 'B':
            h_activated = self.horizontal_output(h_activated)
            h_activated = h + h_activated

        return [v_out, h_activated]


def quantise(images, q_levels):
    """Quantise image into q levels"""
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
    q_levels = 256
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
    # Create encoder model
    inputs = keras.layers.Input(shape=(height, width, n_channel))

    x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.MaxPool2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPool2D((2, 2), padding='same')(x)
    x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = keras.layers.MaxPool2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    decoded = keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = tf.keras.Model(inputs, decoded)
    encoder = tf.keras.Model(inputs, encoded)


    # https://github.com/RishabGoel/PixelCNN/blob/master/pixel_cnn.py
    # https://github.com/jonathanventura/pixelcnn/blob/master/pixelcnn.py
    n_channel = 1

    inputs = keras.layers.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(filters=100, kernel_size=5, strides=1, activation='relu')(inputs)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = keras.layers.Conv2D(filters=150, kernel_size=5, strides=1, activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = keras.layers.Conv2D(filters=200, kernel_size=3, strides=1, activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10, activation='linear')(x)

    labels = keras.layers.Input(shape=(1,), dtype=tf.int32)

    y_labels = tf.broadcast_to(tf.expand_dims(tf.expand_dims(labels, -1), -1), [batch_size, 28, 28, 1])
    y_one = tf.one_hot(tf.cast(y_labels, tf.int32), depth=10)
    y_one = tf.squeeze(y_one)

    y_ae = tf.broadcast_to(tf.expand_dims(tf.expand_dims(x, 1), 1), [batch_size, 28, 28, 10])

    y = tf.concat((y_one, y_ae), axis=-1)

    x = MaskedConv2D(mask_type='A', filters=256, kernel_size=7, strides=1)(inputs)
    for i in range(7):
        x = gated_block_pass(n_filters=128, kernel_size=3, input_tensor=x, y=y)
    v, h = tf.split(x, 2, axis=-1)
    x = keras.layers.Activation(activation='relu')(h)
    x = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1)(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1)(x)
    x = keras.layers.Conv2D(filters=n_channel * q_levels, kernel_size=1, strides=1)(x)  # shape [N,H,W,DC]

    pixelcnn = tf.keras.Model(inputs=[inputs, labels], outputs=x)

    learning_rate = 3e-4
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    compute_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_step(bla):
        batch_x, batch_y, batch_target = bla
        with tf.GradientTape() as ae_tape:
            logits = pixelcnn([batch_x, batch_target])

            logits = tf.reshape(logits, [-1, 28, 28, q_levels, n_channel])  # shape [N,H,W,DC] -> [N,H,W,D,C]
            logits = tf.transpose(logits, perm=[0, 1, 2, 4, 3])  # shape [N,H,W,D,C] -> [N,H,W,C,D]

            loss = compute_loss(tf.one_hot(batch_y, q_levels), logits)

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
