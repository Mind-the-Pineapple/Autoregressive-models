import random as rn

import tensorflow as tf
from tensorflow import keras
import numpy as np


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

        self.conv2a = MaskedConv2D(mask_type='B', filters=h, kernel_size=1, strides=1)
        self.conv2b = MaskedConv2D(mask_type='B', filters=h, kernel_size=3, strides=1)
        self.conv2c = MaskedConv2D(mask_type='B', filters=2*h, kernel_size=1, strides=1)

    def call(self, input_tensor):
        x = tf.nn.relu(input_tensor)
        x = self.conv2a(x)

        x = tf.nn.relu(x)
        x = self.conv2b(x)

        x = tf.nn.relu(x)
        x = self.conv2c(x)

        x += input_tensor
        return x


def binarize(images):
    """
    Stochastically binarize values in [0, 1] by treating them as p-values of
    a Bernoulli distribution.
    """
    return (np.random.uniform(size=images.shape) < images).astype('float32')

def binarize(x):
    x[x >= .5] = 1.
    x[x < .5] = 0.
    return x


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
    x_train = binarize(x_train)
    x_test = binarize(x_test)

    batch_size = 100
    train_buf = 60000

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.shuffle(buffer_size=train_buf)
    train_dataset = train_dataset.batch(batch_size)

    # https://github.com/RishabGoel/PixelCNN/blob/master/pixel_cnn.py
    # https://github.com/jonathanventura/pixelcnn/blob/master/pixelcnn.py
    n_channel = 1
    discrete_channel = 2

    inputs = keras.layers.Input(shape=(28, 28, 1))
    x = MaskedConv2D(mask_type='A', filters=128, kernel_size=7, strides=1)(inputs)
    for i in range(15):
        x = ResidualBlock(h=64)(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = MaskedConv2D(mask_type='B', filters=256, kernel_size=1, strides=1)(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = MaskedConv2D(mask_type='B', filters=n_channel * discrete_channel, kernel_size=1, strides=1)(x)
    # x = keras.layers.Activation(activation='sigmoid')(x)

    pixelcnn = tf.keras.Model(inputs=inputs, outputs=x)

    learning_rate = 3e-4
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    @tf.function
    def train_step(x):
        with tf.GradientTape() as ae_tape:
            z = pixelcnn(x)

            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=x, name='loss'))
        ae_grads = ae_tape.gradient(loss, pixelcnn.trainable_variables)
        optimizer.apply_gradients(zip(ae_grads, pixelcnn.trainable_variables))

        return loss

    epochs = 10
    for epoch in range(epochs):
        print(epoch)
        for x in train_dataset:
            loss = train_step(x)
            print(loss)

    with tf.device('/GPU:1'):

        samples = np.zeros((1, 28, 28, 1), dtype='float32')

        for i in range(28):
            for j in range(28):
                print("{} {}".format(i, j))
                A = pixelcnn(samples)
                next_sample = binarize(A.numpy())
                print(next_sample[:, i, j, 0])
                samples[:, i, j, 0] = next_sample[:, i, j, 0]



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





# https://github.com/igul222/pixel_rnn/blob/master/pixel_rnn.py
cost = T.mean(T.nnet.binary_crossentropy(output, inputs))

params = lib.search(cost, lambda x: hasattr(x, 'param'))
lib.utils.print_params_info(params)

grads = T.grad(cost, wrt=params, disconnected_inputs='warn')
grads = [T.clip(g, lib.floatX(-GRAD_CLIP), lib.floatX(GRAD_CLIP)) for g in grads]









# https://github.com/Joluo/PixelRNN/blob/master/tf/pixel_rnn.py
logits = conv2d(output, hidden_dim, 1, 1, 'b', 'output_conv3')
self.output = tf.nn.sigmoid(logits)

# self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
#    logits=logits, labels=self.inputs, name='loss'))
# self.loss = tf.reduce_mean(tf.contrib.keras.metrics.binary_crossentropy(self.inputs, self.output))
self.loss = tf.reduce_mean(
    -(tf.multiply(self.inputs, tf.log(self.output)) + tf.multiply(1 - self.inputs, tf.log(1 - self.output))))


# https://github.com/singh-hrituraj/PixelCNN-Pytorch

#
#     new_grads_and_vars = \
#         [(tf.clip_by_value(gv[0], -conf.grad_clip, conf.grad_clip), gv[1]) for gv in grads_and_vars]
# self.optim = optimizer.apply_gradients(new_grads_and_vars)

optimizer = tf.train.RMSPropOptimizer(1e-3)
grads_and_vars = optimizer.compute_gradients(loss)

new_grads_and_vars = [(tf.clip_by_value(gv[0], -1, 1), gv[1]) for gv in grads_and_vars]
optim = optimizer.apply_gradients(new_grads_and_vars)