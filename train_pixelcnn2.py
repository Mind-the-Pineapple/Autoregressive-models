"""
https://github.com/igul222/pixel_rnn/blob/master/pixel_rnn.py
https://github.com/rampage644/wavenet/blob/master/wavenet/models.py
https://github.com/jakebelew/gated-pixel-cnn/blob/master/network.py
"""
import random as rn

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class MaskedConv2D(tf.keras.layers.Layer):

    # Mask
    #         -------------------------------------
    #        |  1       1       1       1       1 |
    #        |  1       1       1       1       1 |
    #        |  1       1    1 if B     0       0 |   H // 2
    #        |  0       0       0       0       0 |   H // 2 + 1
    #        |  0       0       0       0       0 |
    #         -------------------------------------
    #  index    0       1     W//2    W//2+1
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

        self.conv2a = keras.layers.Conv2D(filters=h, kernel_size=1, strides=1)
        self.conv2b = MaskedConv2D(mask_type='B', filters=h, kernel_size=3, strides=1)
        self.conv2c = keras.layers.Conv2D(filters=2*h, kernel_size=1, strides=1)

    def call(self, input_tensor):
        x = tf.nn.relu(input_tensor)
        x = self.conv2a(x)

        x = tf.nn.relu(x)
        x = self.conv2b(x)

        x = tf.nn.relu(x)
        x = self.conv2c(x)

        x += input_tensor
        return x

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

    q_levels = 2
    x_train_quantised = quantisize(x_train,q_levels)
    x_test_quantised = quantisize(x_test,q_levels)

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

    inputs = keras.layers.Input(shape=(28, 28, 1))
    x = MaskedConv2D(mask_type='A', filters=128, kernel_size=7, strides=1)(inputs)
    for i in range(15):
        x = ResidualBlock(h=64)(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1)(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(filters=n_channel * q_levels, kernel_size=1, strides=1)(x)  # shape [N,H,W,DC]

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

        gradients = ae_tape.gradient(loss, pixelcnn.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(zip(gradients, pixelcnn.trainable_variables))

        return loss

    epochs = 50
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
            B = tf.nn.softmax(A[:,i,j,:], axis=-1)
            print(B[:,i,j,1].numpy())
            next_sample = binarize(B[:,:,:,1].numpy())
            print(next_sample[:, i, j])
            samples[:, i, j, 0] = next_sample[:, i, j]


# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------


 clipped_gradients = [(tf.clip_by_value(_[0],-1,1), _[1]) for _ in gradients]

def preprocess(q_levels):
    def preprocess_fcn(images, labels):
        # Create the target pixels from the image. Quantize the scalar pixel values into q_level indices.
        target_pixels = np.clip(((images * q_levels).astype('int64')), 0, q_levels - 1)  # [N,H,W,C]
        return (images, target_pixels)

    return preprocess_fcn



#
# https://github.com/RishabGoel/PixelCNN/blob/master/pixel_cnn.py
# https://github.com/jonathanventura/pixelcnn/blob/master/pixelcnn.py
"""Combined layer"""
combined_stack = Conv2D(x_h, feature_maps, feature_maps, [1, 1, feature_maps, feature_maps], border_mode="valid")
if q_levels != None:
    out_dim = input_dim
else:
    out_dim = input_dim * q_levels

combined_stack_final = Conv2D(combined_stack.get_output(), feature_maps, out_dim, [1, 1, feature_maps, out_dim],
                              border_mode="valid")

pre_final_output = tf.traanspose(combined_stack_final.get_output(), perm=[0, 2, 3, 1])
if q_levels:
    old_shape = pre_final_output.get_shape().as_list()
    self.Y = tf.reshape(pre_final_output, shape=[old_shape[0], old_shape[1], old_shape[2] // q_levels, -1])
else:
    self.Y = pre_final_output



https://github.com/rampage644/wavenet/commit/bbe8e3a98e48e3c3687d8ec9e342ef22cf530ac2
nll = F.sigmoid_cross_entropy(y, F.cast(x, 'i'))


# https://github.com/Joluo/PixelRNN/blob/master/tf/pixel_rnn.py
logits = conv2d(output, hidden_dim, 1, 1, 'b', 'output_conv3')
self.output = tf.nn.sigmoid(logits)

# self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#    logits=logits, labels=self.inputs, name='loss'))
# self.loss = tf.reduce_mean(tf.contrib.keras.metrics.binary_crossentropy(self.inputs, self.output))
self.loss = tf.reduce_mean(
    -(tf.multiply(self.inputs, tf.log(self.output)) + tf.multiply(1 - self.inputs, tf.log(1 - self.output))))


# https://github.com/singh-hrituraj/PixelCNN-Pytorch

# https://github.com/rampage644/wavenet/blob/master/wavenet/models.py
nll = F.softmax_cross_entropy(y, t, normalize=True)
chainer.report({'nll': nll, 'bits/dim': nll / dims}, self)




https://github.com/jakebelew/gated-pixel-cnn/blob/master/network.py
if (num_channels > 1):
    self.logits = tf.reshape(self.logits,
                             [-1, height, width, q_levels, num_channels])  # shape [N,H,W,DC] -> [N,H,W,D,C]
    self.logits = tf.transpose(self.logits, perm=[0, 1, 2, 4, 3])  # shape [N,H,W,D,C] -> [N,H,W,C,D]

flattened_logits = tf.reshape(self.logits, [-1, q_levels])  # [N,H,W,C,D] -> [NHWC,D]
target_pixels_loss = tf.reshape(self.target_pixels, [-1])  # [N,H,W,C] -> [NHWC]

logger.info("Building loss and optims")
self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    flattened_logits, target_pixels_loss))

flattened_output = tf.nn.softmax(flattened_logits)  # shape [NHWC,D], values [probability distribution]
self.output = tf.reshape(flattened_output, [-1, height, width, num_channels,
                                            q_levels])  # shape [N,H,W,C,D], values [probability distribution]

optimizer = tf.train.RMSPropOptimizer(conf.learning_rate)
grads_and_vars = optimizer.compute_gradients(self.loss)

new_grads_and_vars = \
    [(tf.clip_by_value(gv[0], -conf.grad_clip, conf.grad_clip), gv[1]) for gv in grads_and_vars]
self.optim = optimizer.apply_gradients(new_grads_and_vars)

show_all_variables()

logger.info("Building gated_pixel_cnn finished")


def predict(self, images):
    '''
    images # shape [N,H,W,C]
    returns predicted image # shape [N,H,W,C]
    '''
    # self.output shape [NHWC,D]
    pixel_value_probabilities = self.sess.run(self.output, {
        self.inputs: images})  # shape [N,H,W,C,D], values [probability distribution]

    # argmax or random draw # [NHWC,1]  quantized index - convert back to pixel value
    pixel_value_indices = np.argmax(pixel_value_probabilities,
                                    4)  # shape [N,H,W,C], values [index of most likely pixel value]
    pixel_values = np.multiply(pixel_value_indices, ((self.pixel_depth - 1) / (self.q_levels - 1)))  # shape [N,H,W,C]

    return pixel_values


def test(self, images, with_update=False):
    if with_update:
        _, cost = self.sess.run([self.optim, self.loss],
                                {self.inputs: images[0], self.target_pixels: images[1]})
    else:
        cost = self.sess.run(self.loss, {self.inputs: images[0], self.target_pixels: images[1]})
    return cost


def generate_from_occluded(self, images, num_generated_images, occlude_start_row):
    samples = np.copy(images[0:num_generated_images, :, :, :])
    samples[:, occlude_start_row:, :, :] = 0.

    for i in xrange(occlude_start_row, self.height):
        for j in xrange(self.width):
            for k in xrange(self.channel):
                next_sample = self.predict(samples) / (self.pixel_depth - 1.)  # argmax or random draw here
                samples[:, i, j, k] = next_sample[:, i, j, k]

    return samples


def generate(self, images):
    samples = images[0:9, :, :, :]
    occlude_start_row = 18
    samples[:, occlude_start_row:, :, :] = 0.

    for i in xrange(occlude_start_row, self.height):
        for j in xrange(self.width):
            for k in xrange(self.channel):
                next_sample = self.predict(samples) / (self.pixel_depth - 1.)  # argmax or random draw here
                samples[:, i, j, k] = next_sample[:, i, j, k]


return samples