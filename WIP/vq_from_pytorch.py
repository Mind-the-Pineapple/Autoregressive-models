"""https://github.com/rosinality/vq-vae-2-pytorch"""
import tensorflow as tf
from tensorflow import keras


class ResBlock(tf.keras.Model):
  def __init__(self, filters):
    super(ResBlock, self).__init__(name='')
    filters1, filters2 = filters

    self.act_a = keras.layers.Activation(activation='relu')

    self.conv2a = keras.layers.Conv2D(filters=filters1,
                                      kernel_size=(3,3),
                                      strides=(1, 1),
                                      padding='same',
                                      activation='linear')

    self.act_b = keras.layers.Activation(activation='relu')

    self.conv2b = keras.layers.Conv2D(filters=filters2,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding='same',
                                      activation='linear')


  def call(self, input_tensor, training=False):
    x = self.act_a(input_tensor)
    x = self.conv2a(x, training=training)
    x = self.act_b(x)
    x = self.conv2b(x, training=training)
    x += input_tensor

    return x


class Encoder(tf.keras.Model):
  def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
    super(Encoder, self).__init__(name='')

    if stride == 4:
        blocks = [keras.layers.Conv2D(filters=filters1,
                                      kernel_size=(3,3),
                                      strides=(1, 1),
                                      padding='same',
                                      activation='linear')]

    self.act_a = keras.layers.Activation(activation='relu')

    self.conv2a = keras.layers.Conv2D(filters=in_channel,
                                      kernel_size=(3,3),
                                      strides=(1, 1),
                                      padding='same',
                                      activation='linear')

    self.act_b = keras.layers.Activation(activation='relu')

    self.conv2b = keras.layers.Conv2D(filters=in_channel,
                                      kernel_size=(1, 1),
                                      strides=(1, 1),
                                      padding='same',
                                      activation='linear')


  def call(self, input_tensor, training=False):
    x = self.act_a(input_tensor)
    x = self.conv2a(x, training=training)
    x = self.act_b(x)
    x = self.conv2b(x, training=training)
    x += input_tensor

    return x