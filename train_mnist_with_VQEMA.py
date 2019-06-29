"""

Refs:
https://github.com/deepmind/sonnet/blob/master/sonnet/examples/vqvae_example.ipynb
https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py
https://github.com/rstudio/keras/blob/master/vignettes/examples/vq_vae.R
https://blogs.rstudio.com/tensorflow/posts/2019-01-24-vq-vae/

https://nbviewer.jupyter.org/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
"""
import time
import random as rn

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.python.training import moving_averages

# def main():

# ---------------------------------------------------------------------------------------------------------------
random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)
rn.seed(random_seed)
# ---------------------------------------------------------------------------------------------------------------

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


x_train = (x_train.astype('float32') / 255.) - 0.5
x_test = (x_test.astype('float32') / 255.) - 0.5

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

data_variance = np.var(x_train)
# ---------------------------------------------------------------------------------------------------------------

batch_size = 32
train_buf = 100

train_dataset = tf.data.Dataset.from_tensor_slices((x_train))
train_dataset = train_dataset.shuffle(buffer_size=train_buf)
train_dataset = train_dataset.batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test))
test_dataset = test_dataset.batch(batch_size)

# ---------------------------------------------------------------------------------------------------------------
input_shape = (28, 28, 1)
embedding_dim = 64
decoder_input_shape = (7, 7, embedding_dim)

# ---------------------------------------------------------------------------------------------------------------
def residual_stack(h, num_hiddens, num_residual_layers, num_residual_hiddens):
    for i in range(num_residual_layers):
        h_i = keras.layers.Activation(activation='relu')(h)

        h_i = keras.layers.Conv2D(filters=num_residual_hiddens,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  padding='same',
                                  name="res3x3_%d" % i)(h_i)

        h_i = keras.layers.Activation(activation='relu')(h_i)

        h_i = keras.layers.Conv2D(filters=num_hiddens,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  padding='same',
                                  name="res1x1_%d" % i)(h_i)

        h += h_i

    return keras.layers.Activation(activation='relu')(h)


def create_encoder(num_hiddens, num_residual_layers, num_residual_hiddens):
    inputs = keras.layers.Input(shape=input_shape)

    x = keras.layers.Conv2D(filters=int(num_hiddens / 2),
                            kernel_size=(4, 4),
                            strides=(2, 2),
                            padding='same',
                            activation='relu')(inputs)

    x = keras.layers.Conv2D(filters=num_hiddens,
                            kernel_size=(4, 4),
                            strides=(2, 2),
                            padding='same',
                            activation='relu')(x)

    x = keras.layers.Conv2D(filters=num_hiddens,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='relu')(x)

    x = residual_stack(x, num_hiddens, num_residual_layers, num_residual_hiddens)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


def create_decoder(num_hiddens, num_residual_layers, num_residual_hiddens):
    inputs = keras.layers.Input(shape=decoder_input_shape)

    x = keras.layers.Conv2D(filters=num_hiddens,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding='same',
                            activation='linear')(inputs)

    x = residual_stack(x, num_hiddens, num_residual_layers, num_residual_hiddens)

    x = keras.layers.Conv2DTranspose(filters=int(num_hiddens / 2),
                                    kernel_size=(4, 4),
                                    strides=(2, 2),
                                    padding='same',
                                    activation='relu')(x)

    x = keras.layers.Conv2DTranspose(filters=3,
                            kernel_size=(4, 4),
                            strides=(2, 2),
                            padding='same',
                            activation='linear')(x)


    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

# ---------------------------------------------------------------------------------------------------------------
batch_size = 32
image_size = 32
num_training_updates = 50000
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
embedding_dim = 64
num_embeddings = 512
commitment_cost = 0.25
decay = 0.99
learning_rate = 3e-4

encoder = create_encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
decoder = create_decoder(num_hiddens, num_residual_layers, num_residual_hiddens)

pre_vq_conv1 = keras.layers.Conv2D(filters=embedding_dim,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        padding='same',
                        activation='linear')

class VectorQuantizerEMA():
    def __init__(self, embedding_dim, num_embeddings, commitment_cost, decay, epsilon=1e-5, name='VectorQuantizerEMA'):
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._decay = decay
        self._commitment_cost = commitment_cost
        self._epsilon = epsilon

        initializer = tf.random_normal_initializer()
        self._w = tf.Variable(initializer((embedding_dim, num_embeddings)), name='embedding')
        self._ema_cluster_size = tf.Variable(tf.constant_initializer(0.0)((num_embeddings)), name='ema_cluster_size')
        self._ema_w = tf.Variable(self._w.read_value(),name='ema_dw')

    def _build(self, inputs, training=False):
        w = self._w.read_value()

        flat_inputs = tf.reshape(inputs, [-1, self._embedding_dim])
        distances = (tf.reduce_sum(flat_inputs ** 2, 1, keepdims=True)
                     - 2 * tf.matmul(flat_inputs, w)
                     + tf.reduce_sum(w ** 2, 0, keepdims=True))

        encoding_indices = tf.argmax(- distances, 1)
        encodings = tf.one_hot(encoding_indices, self._num_embeddings)
        encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1])
        quantized = self.quantize(encoding_indices)
        e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)

        loss = self._commitment_cost * e_latent_loss
        quantized = inputs + tf.stop_gradient(quantized - inputs)
        avg_probs = tf.reduce_mean(encodings, 0)
        perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))

        return {'quantize': quantized,
                'loss': loss,
                'perplexity': perplexity,
                'encodings': encodings,
                'encoding_indices': encoding_indices, }

    @property
    def embeddings(self):
        return self._w

    def quantize(self, encoding_indices):
        w = tf.transpose(self.embeddings.read_value(), [1, 0])
        return tf.nn.embedding_lookup(w, encoding_indices)

    def update_table(self, inputs, encodings):
        flat_inputs = tf.reshape(inputs, [-1, self._embedding_dim])

        updated_ema_cluster_size = moving_averages.assign_moving_average(self._ema_cluster_size,
                                                                         tf.reduce_sum(encodings, 0), self._decay)
        dw = tf.matmul(flat_inputs, encodings, transpose_a=True)
        updated_ema_w = moving_averages.assign_moving_average(self._ema_w, dw, self._decay)
        n = tf.reduce_sum(updated_ema_cluster_size)
        updated_ema_cluster_size = (
                    (updated_ema_cluster_size + self._epsilon) / (n + self._num_embeddings * self._epsilon) * n)

        normalised_updated_ema_w = (updated_ema_w / tf.reshape(updated_ema_cluster_size, [1, -1]))

        self._w.assign(normalised_updated_ema_w)


# Vector quantizer -------------------------------------------------------------------
vq_vae = VectorQuantizerEMA(
      embedding_dim=embedding_dim,
      num_embeddings=num_embeddings,
      commitment_cost=commitment_cost,
      decay=decay)

optimizer = tf.keras.optimizers.Adam(lr=learning_rate)


@tf.function
def train_step(x):
    with tf.GradientTape(persistent=True) as ae_tape:
        z = pre_vq_conv1(encoder(x))

        vq_output_train = vq_vae._build(z, training=True)
        x_recon = decoder(vq_output_train["quantize"])
        recon_error = tf.reduce_mean((x_recon - x) ** 2) / data_variance  # Normalized MSE
        loss = recon_error + vq_output_train["loss"]

        perplexity = vq_output_train["perplexity"]

    ae_grads = ae_tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables+ pre_vq_conv1.trainable_variables)
    optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables+ pre_vq_conv1.trainable_variables))

    return recon_error, perplexity, z, vq_output_train['encodings']


train_res_recon_error = []
train_res_perplexity = []
epochs = 100
iteraction = 0

for epoch in range(epochs):
    start = time.time()

    total_loss = 0.0
    total_per = 0.0
    num_batches = 0

    for x in train_dataset:
        recon_error, perplexity, z, encodings = train_step(x)
        vq_vae.update_table(z, encodings)
        total_loss += recon_error
        total_per += perplexity

        num_batches += 1
        iteraction += 1

        train_res_recon_error.append(recon_error)
        train_res_perplexity.append(perplexity)

        if (iteraction + 1) % 100 == 0:
            print('%d iterations' % (iteraction + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()


    train_loss = total_loss / num_batches
    total_per = total_per / num_batches

    train_res_recon_error.append(train_loss)
    train_res_perplexity.append(total_per)

    # print(vq_vae.embeddings)

    epoch_time = time.time() - start
    #
    # template = ("{:4d}: TIME: {:.2f} ETA: {:.2f} AE_LOSS: {:.4f} PER: {:.4f} ")
    # print(template.format(epoch + 1,
    #                       epoch_time, epoch_time * (epochs - epoch),
    #                       train_loss, total_per))


import matplotlib.pyplot as plt
f = plt.figure(figsize=(16,8))
ax = f.add_subplot(1,2,1)
ax.plot(train_res_recon_error)
ax.set_yscale('log')
ax.set_title('NMSE.')

ax = f.add_subplot(1,2,2)
ax.plot(train_res_perplexity)
ax.set_title('Average codebook usage (perplexity).')


def convert_batch_to_image_grid(image_batch):
    reshaped = tf.reshape(image_batch, (4, 8, 28, 28, 1))
    reshaped = tf.transpose(reshaped, perm=(0, 2, 1, 3, 4))
    reshaped = tf.reshape(reshaped, (4 * 28, 8 * 28, 1))
    return reshaped + 0.5

original_train = next(iter(train_dataset))
z = pre_vq_conv1(encoder(original_train))
vq_output_train = vq_vae._build(z, training=True)
x_recon_train = decoder(vq_output_train["quantize"])

f = plt.figure(figsize=(16,8))
ax = f.add_subplot(2,2,1)
ax.imshow(convert_batch_to_image_grid(original_train)[:,:,0],
          interpolation='nearest')
ax.set_title('training data originals')
plt.axis('off')

ax = f.add_subplot(2,2,2)
ax.imshow(convert_batch_to_image_grid(x_recon_train)[:,:,0],
          interpolation='nearest')
ax.set_title('training data reconstructions')
plt.axis('off')

original_test = next(iter(test_dataset))
z = pre_vq_conv1(encoder(original_test))
vq_output_train = vq_vae._build(z, training=True)
x_recon_test = decoder(vq_output_train["quantize"])

ax = f.add_subplot(2,2,3)
ax.imshow(convert_batch_to_image_grid(original_test)[:,:,0],
          interpolation='nearest')
ax.set_title('validation data originals')
plt.axis('off')

ax = f.add_subplot(2,2,4)
ax.imshow(convert_batch_to_image_grid(x_recon_test)[:,:,0],
          interpolation='nearest')
ax.set_title('validation data reconstructions')
plt.axis('off')
