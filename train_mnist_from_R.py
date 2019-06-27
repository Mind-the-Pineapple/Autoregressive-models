""""""
import random as rn

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

from tensorflow.python.training import moving_averages

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

    batch_size = 64
    train_buf = 60000

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train))
    train_dataset = train_dataset.shuffle(buffer_size=train_buf)
    train_dataset = train_dataset.batch(batch_size)

    learning_rate = 0.001
    latent_size = 1
    num_codes = 64
    code_size = 16
    base_depth = 32
    activation = "elu"
    beta =0.25
    decay = 0.99
    input_shape =(28, 28, 1)

    # ---------------------------------------------------------------------------------------------------------------
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(filters=base_depth, kernel_size=5, padding='same', activation='elu')(inputs)
    x = keras.layers.Conv2D(filters=base_depth, kernel_size=5, strides=2, padding='same', activation='elu')(x)
    x = keras.layers.Conv2D(filters=2*base_depth, kernel_size=5, padding='same', activation='elu')(x)
    x = keras.layers.Conv2D(filters=2*base_depth, kernel_size=5, strides=2, padding='same', activation='elu')(x)
    x = keras.layers.Conv2D(filters=4*latent_size, kernel_size=7, padding='valid', activation='elu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=latent_size*code_size)(x)
    x = keras.layers.Reshape((latent_size, code_size))(x)
    encoder = tf.keras.Model(inputs=inputs, outputs=x)

    # ---------------------------------------------------------------------------------------------------------------
    inputs = keras.layers.Input(shape=(latent_size, code_size))
    x = keras.layers.Reshape((1 , latent_size, code_size))(inputs)
    x = keras.layers.Conv2DTranspose(filters=2 * base_depth, kernel_size=7, padding='valid', activation='elu')(x)
    x = keras.layers.Conv2DTranspose(filters=2 * base_depth, kernel_size=5, padding='same', activation='elu')(x)
    x = keras.layers.Conv2DTranspose(filters=2 * base_depth, kernel_size=5, strides=2, padding='same', activation='elu')(x)
    x = keras.layers.Conv2DTranspose(filters=base_depth, kernel_size=5,  padding='same', activation='elu')(x)
    x = keras.layers.Conv2DTranspose(filters=base_depth, kernel_size=5, strides=2, padding='same', activation='elu')(x)
    x = keras.layers.Conv2DTranspose(filters=base_depth, kernel_size=5, padding='same', activation='elu')(x)
    x = keras.layers.Conv2D(filters=1, kernel_size=5, padding='same', activation='linear')(x)
    decoder = tf.keras.Model(inputs=inputs, outputs=x)

    # Vector quantizer -------------------------------------------------------------------

    initializer = tf.initializers.GlorotUniform()
    init = tf.initializers.Constant(0)
    codebook = tf.Variable(initializer(shape=(num_codes, code_size)), dtype=tf.float32)
    ema_count = tf.Variable(init(shape=(num_codes)), trainable=False)
    ema_means = tf.Variable(codebook.read_value(), trainable=False)

    def vector_quantizer(x):
        distances = tf.norm(tf.expand_dims(x, axis=2) - tf.reshape(codebook, (1,1,num_codes, code_size)), axis=3)
        assignments = tf.argmin(distances, axis=2)
        one_hot_assignments = tf.one_hot(assignments, depth=num_codes)
        nearest_codebook_entries = tf.reduce_sum(tf.expand_dims(one_hot_assignments, -1) *  tf.reshape(codebook, (1, 1, num_codes, code_size)), axis = 2)

        return nearest_codebook_entries, one_hot_assignments

    def update_ema(codes):
        updated_ema_count = moving_averages.assign_moving_average(ema_count, tf.reduce_sum(one_hot_assignments, axis=(0, 1)), decay, zero_debias=False)
        updated_ema_means = moving_averages.assign_moving_average(ema_means,
                                                                  tf.reduce_sum(tf.expand_dims(codes, 2)*tf.expand_dims(one_hot_assignments, axis=(0,1)), decay, zero_debias=False))

        # Add small value to avoid dividing by zero
        updated_ema_count = updated_ema_count + 1e-5
        updated_ema_means = updated_ema_means / tf.expand_dims(updated_ema_count, axis=-1)

        tf.assign(codebook, updated_ema_means)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    @tf.function
    def train_step(x):

        with tf.GradientTape(persistent=True) as ae_tape:
            codes = encoder(x)
            nearest_codebook_entries, one_hot_assignments = vector_quantizer(codes)
            codes_straight_through = codes + tf.stop_gradient(nearest_codebook_entries - codes)
            decoder_distribution = decoder(codes_straight_through)

            reconstruction_loss = -tf.reduce_mean(decoder_distribution.log_prob(x))

            commitment_loss = tf.reduce_mean(tf.square(codes - tf.stop_gradient(nearest_codebook_entries)))

            prior_dist = tfd.Multinomial(total_count=1, logits=tf.zeros((latent_size, num_codes)))
            prior_loss = -tf.reduce_mean(tf.reduce_sum(prior_dist.log_prob(one_hot_assignments), 1))

            loss = reconstruction_loss + beta * commitment_loss + prior_loss

        ae_grads = ae_tape.gradient(loss, encoder.trainable_variables + decoder.trainable_variables)
        optimizer.apply_gradients(zip(ae_grads, encoder.trainable_variables + decoder.trainable_variables))

        update_ema(vector_quantizer,
                   one_hot_assignments,
                   codes,
                   decay)

        total_loss = total_loss + loss
        reconstruction_loss_total = reconstruction_loss_total + reconstruction_loss
        commitment_loss_total = commitment_loss_total + commitment_loss
        prior_loss_total = prior_loss_total + prior_loss

    epochs = 20
    for epoch in range(epochs):
        total_loss =0
        reconstruction_loss_total =0
        commitment_loss_total =0
        prior_loss_total =0

        total_loss = 0.0
        num_batches = 0
        for x in train_dataset:
            total_loss += train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches








    # tfd.Independent(tfd.Bernoulli(logits=x), reinterpreted_batch_ndims = len(input_shape))
    #
    # tfp.layers.DistributionLambda(
    #     lambda t: tfd.Normal(loc=t[..., :1],
    #                          scale=1e-3 + tf.math.softplus(0.05 * t[..., 1:]))),

    # TODO: Quantification part

    # This value is not that important, usually 64 works.
    # This will not change the capacity in the information-bottleneck.
    embedding_dim = 64

    # The higher this value, the higher the capacity in the information bottleneck.
    num_embeddings = 512
    commitment_cost = 0.25

    pre_vq_conv1 = keras.layers.Conv2D(filters=embedding_dim,
                              kernel_size=(1, 1),
                              strides=(1, 1),
                              name="to_vq")

    flat_inputs = tf.reshape(C, [-1, embedding_dim])

    _w = tf.Variable(tf.random.uniform([embedding_dim, num_embeddings]), dtype=tf.float32)

    distances = (tf.reduce_sum(flat_inputs**2, 1, keepdims=True) - 2 * tf.matmul(flat_inputs, _w) + tf.reduce_sum(_w ** 2, 0, keepdims=True))
    encoding_indices = tf.argmax(- distances, 1)
    encodings = tf.one_hot(encoding_indices, num_embeddings)
    encoding_indices = tf.reshape(encoding_indices, tf.shape(C)[:-1])

    w = tf.transpose(_w.read_value(), [1, 0])
    quantized =  tf.nn.embedding_lookup(w, encoding_indices)

    e_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - C) ** 2)
    q_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(C)) ** 2)
    vq_loss = q_latent_loss + commitment_cost * e_latent_loss

    quantized = C + tf.stop_gradient(quantized - C)
    avg_probs = tf.reduce_mean(encodings, 0)
    perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10)))

    x_recon = decoder(quantized)
    recon_error = tf.reduce_mean((x_recon - x) ** 2)
    loss = recon_error + vq_loss




    self.recon = tf.reduce_mean((self.p_x_z - x) ** 2, axis=[0, 1, 2, 3])
    self.vq = tf.reduce_mean(
        tf.norm(tf.stop_gradient(self.z_e) - z_q, axis=-1) ** 2,
        axis=[0, 1, 2])
    self.commit = tf.reduce_mean(
        tf.norm(self.z_e - tf.stop_gradient(z_q), axis=-1) ** 2,
        axis=[0, 1, 2])
    self.loss = self.recon + self.vq + beta * self.commit




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