""""""
import random as rn

import numpy as np
import tensorflow as tf
from tensorflow import keras
# import tensorflow_datasets as tfds

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

    inputs = keras.layers.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='linear')(inputs)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='linear')(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='linear')(x)
    x = keras.layers.Activation(activation='relu')(x)
    encoder = tf.keras.Model(inputs=inputs, outputs=x)

    inputs = keras.layers.Input(shape=(4, 4, 64))
    x = keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='linear')(inputs)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='linear')(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='linear')(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', activation='linear')(x)
    x = keras.layers.Activation(activation='sigmoid')(x)
    x = keras.layers.Cropping2D(cropping=((2, 2), (2, 2)))(x)
    decoder = tf.keras.Model(inputs=inputs, outputs=x)

    # TODO: Quantification part



    with tf.variable_scope('embed') :
        embeds = tf.get_variable('embed', [K,D])

    self.recon = tf.reduce_mean((self.p_x_z - x) ** 2, axis=[0, 1, 2, 3])
    self.vq = tf.reduce_mean(
        tf.norm(tf.stop_gradient(self.z_e) - z_q, axis=-1) ** 2,
        axis=[0, 1, 2])
    self.commit = tf.reduce_mean(
        tf.norm(self.z_e - tf.stop_gradient(z_q), axis=-1) ** 2,
        axis=[0, 1, 2])
    self.loss = self.recon + self.vq + beta * self.commit

    # NLL
    # TODO: is it correct impl?
    # it seems tf.reduce_prod(tf.shape(self.z_q)[1:2]) should be multipled
    # in front of log(1/K) if we assume uniform prior on z.
    self.nll = -1. * (tf.reduce_mean(tf.log(self.p_x_z), axis=[1, 2, 3]) + tf.log(1 / tf.cast(K, tf.float32))) / tf.log(
        2.)

      # SONNET

      self._w = tf.get_variable('embedding', [embedding_dim, num_embeddings],
initializer=initializer, trainable=True)


    flat_inputs = tf.reshape(inputs, [-1, self._embedding_dim])


    distances = (tf.reduce_sum(flat_inputs**2, 1, keepdims=True) - 2 * tf.matmul(flat_inputs, self._w) + tf.reduce_sum(self._w ** 2, 0, keepdims=True))


    encoding_indices = tf.argmax(- distances, 1)
    encodings = tf.one_hot(encoding_indices, self._num_embeddings)
    encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
    quantized = self.quantize(encoding_indices)



    e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
    q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
    loss = q_latent_loss + self._commitment_cost * e_latent_loss

    quantized = inputs + tf.stop_gradient(quantized - inputs)
    avg_probs = tf.reduce_mean(encodings, 0)
    perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.log(avg_probs + 1e-10)))


  def quantize(self, encoding_indices):
    with tf.control_dependencies([encoding_indices]):
      w = tf.transpose(self.embeddings.read_value(), [1, 0])
return tf.nn.embedding_lookup(w, encoding_indices, validate_indices=False)



def vq_loss(inputs, embedded, commitment=0.25):
    """
    Compute the codebook and commitment losses for an
    input-output pair from a VQ layer.
    """
    return (torch.mean(torch.pow(inputs.detach() - embedded, 2)) +
            commitment * torch.mean(torch.pow(inputs - embedded.detach(), 2)))



def vq(z_e):
    '''Vector Quantization.
    Args:
      z_e: encoded variable. [B, t, D].
    Returns:
      z_q: nearest embeddings. [B, t, D].
    '''
    with tf.variable_scope("vq"):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[hp.K, hp.D],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        z = tf.expand_dims(z_e, -2) # (B, t, 1, D)
        lookup_table_ = tf.reshape(lookup_table, [1, 1, hp.K, hp.D]) # (1, 1, K, D)
        dist = tf.norm(z - lookup_table_, axis=-1) # Broadcasting -> (B, T', K)
        k = tf.argmin(dist, axis=-1) # (B, t)
        z_q = tf.gather(lookup_table, k) # (B, t, D)

return z_q