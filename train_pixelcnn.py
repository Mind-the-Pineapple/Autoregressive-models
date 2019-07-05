import random as rn

import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.python.ops import nn


class MaskedConv2D(tf.keras.layers.Conv2D):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2D, self).__init__(*args, **kwargs)

        assert mask_type in {'A', 'B'}
        self.mask_type = mask_type
        self.mask = None

    def build(self, input_shape):
        super(MaskedConv2D, self).build(input_shape)

        kH, kW = self.kernel_size
        mask = np.ones(self.kernel.shape, dtype=np.float32)
        mask[kH // 2, kW // 2 + (self.mask_type == 'B'):, :, :] = 0.
        mask[kH // 2 + 1:, :, :] = 0.
        self.mask = mask

    def call(self, inputs):
        outputs = self._convolution_op(inputs, tf.math.multiply(self.kernel, self.mask))

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


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
    x_train[x_train >= .5] = 1.
    x_train[x_train < .5] = 0.
    x_test[x_test >= .5] = 1.
    x_test[x_test < .5] = 0.

    batch_size = 10
    train_buf = 60000

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=train_buf)
    train_dataset = train_dataset.batch(batch_size)

    inputs = keras.layers.Input(shape=(28, 28, 1))
    x = MaskedConv2D(mask_type='A', filters=16, kernel_size=7, strides=1, padding='same', activation='linear')(inputs)
    x = keras.layers.Activation(activation='relu')(x)
    x = MaskedConv2D(mask_type='B', filters=32, kernel_size=3, strides=1, padding='same', activation='linear')(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = MaskedConv2D(mask_type='B', filters=64, kernel_size=3, strides=1, padding='same', activation='linear')(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = MaskedConv2D(mask_type='B', filters=64, kernel_size=3, strides=1, padding='same', activation='linear')(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = MaskedConv2D(mask_type='B', filters=1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    encoder = tf.keras.Model(inputs=inputs, outputs=x)

    learning_rate=3e-4
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    # @tf.function
    def train_step(x):
        with tf.GradientTape(persistent=True) as ae_tape:
            z = encoder(x)

            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=x, name='loss'))
        ae_grads = ae_tape.gradient(loss, encoder.trainable_variables)
        optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables))

        return loss


    epochs = 100
    for epoch in range(epochs):

        for x in train_dataset:
            loss = train_step(x)
            print(loss)
