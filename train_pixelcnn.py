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

    batch_size = 10
    train_buf = 60000

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
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



    @tf.function
    def train_step(x):
        with tf.GradientTape(persistent=True) as ae_tape:
            z = encoder(x)

            recon_error = tf.reduce_mean((x_recon - x) ** 2) / data_variance  # Normalized MSE
            loss = recon_error + vq_output_train["loss"]

            perplexity = vq_output_train["perplexity"]

        ae_grads = ae_tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables+ pre_vq_conv1.trainable_variables)
        optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables+ pre_vq_conv1.trainable_variables))

        return recon_error, perplexity, z, vq_output_train['encodings']

epochs = 100
iteraction = 0

for epoch in range(epochs):
    total_loss = 0.0
    total_per = 0.0
    num_batches = 0

    for x in train_dataset:
        print()