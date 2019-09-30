"""
# https://github.com/suga93/pixelcnn_keras/blob/master/core/layers.py
"""
import random as rn
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


class GatedBlock(tf.keras.Model):
    """"""

    def __init__(self, mask_type, filters, kernel_size):
        super(GatedBlock, self).__init__(name='')

        self.mask_type = mask_type
        self.vertical_padding = keras.layers.ZeroPadding2D(padding=((kernel_size//2+1, 0),
                                                                    (kernel_size//2, kernel_size//2)))

        self.vertical_conv = keras.layers.Conv2D(filters=2 * filters,
                                                 kernel_size=[kernel_size//2+1, kernel_size],
                                                 strides=1,
                                                 padding='valid')

        self.vertical_cropping = keras.layers.Cropping2D(cropping=((0, 1), (0, 0)))

        self.horizontal_padding = keras.layers.ZeroPadding2D(padding=((0, 0), (kernel_size // 2+1, 0)))
        self.horizontal_conv = keras.layers.Conv2D(filters=2 * filters,
                                                   kernel_size=[1, kernel_size // 2 + 1],
                                                   strides=1,
                                                   padding='valid')
        if mask_type == 'B':
            self.horizontal_cropping = keras.layers.Cropping2D(cropping=((0, 0), (1, 0)))
        elif mask_type == 'A':
            self.horizontal_cropping = keras.layers.Cropping2D(cropping=((0, 0), (0, 1)))

        self.vertical_to_horizontal_conv = keras.layers.Conv2D(filters=2 * filters, kernel_size=1)

        self.horizontal_output_conv = keras.layers.Conv2D(filters=filters, kernel_size=1)

    def _gate(self, x):
        tanh_preactivation, sigmoid_preactivation = tf.split(x, 2, axis=-1)
        return tf.nn.tanh(tanh_preactivation) * tf.nn.sigmoid(sigmoid_preactivation)

    def call(self, input_tensor):
        v, h = tf.split(input_tensor, 2, axis=-1)

        vertical_preactivation = self.vertical_padding(v)
        vertical_preactivation = self.vertical_conv(vertical_preactivation)
        vertical_preactivation = self.vertical_cropping(vertical_preactivation)

        horizontal_preactivation = self.horizontal_padding(h)
        horizontal_preactivation = self.horizontal_conv(horizontal_preactivation)
        horizontal_preactivation = self.horizontal_cropping(horizontal_preactivation)

        v_to_h = self.vertical_to_horizontal_conv(vertical_preactivation)  # 1x1
        vertical_output = self._gate(vertical_preactivation)

        horizontal_preactivation = horizontal_preactivation + v_to_h
        horizontal_activated = self._gate(horizontal_preactivation)

        if self.mask_type =='B':
            horizontal_activated = self.horizontal_output_conv(horizontal_activated)
            horizontal_activated = h + horizontal_activated

        output = tf.concat((vertical_output, horizontal_activated), axis=-1)
        return output

def quantise(images, q_levels):
    """Quantise image into q levels"""
    return (np.digitize(images, np.arange(q_levels) / q_levels) - 1).astype('float32')


def sample_from(distribution):
    """Sample random values from distribution"""
    batch_size, bins = distribution.shape
    return np.array([np.random.choice(bins, p=distr) for distr in distribution])


# def main():
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

x_train = x_train.reshape(x_train.shape[0], height, width, 1)
x_test = x_test.reshape(x_test.shape[0], height, width, 1)

# --------------------------------------------------------------------------------------------------------------
# Quantise the input data in q levels
q_levels = 16
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
# Create PixelCNN model
# https://github.com/RishabGoel/PixelCNN/blob/master/pixel_cnn.py
# https://github.com/jonathanventura/pixelcnn/blob/master/pixelcnn.py

inputs = keras.layers.Input(shape=(height, width, n_channel))
x = keras.layers.Concatenate()([inputs, inputs])
x = GatedBlock(mask_type='A', filters=64, kernel_size=7)(x)

for i in range(5):
    x = GatedBlock(mask_type='B', filters=64, kernel_size=3)(x)

v, h = tf.split(x, 2, axis=-1)

x = keras.layers.Activation(activation='relu')(h)
x = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1)(x)

x = keras.layers.Activation(activation='relu')(x)
x = keras.layers.Conv2D(filters=n_channel * q_levels, kernel_size=1, strides=1)(x)  # shape [N,H,W,DC]

pixelcnn = tf.keras.Model(inputs=inputs, outputs=x)

# --------------------------------------------------------------------------------------------------------------
# Prepare optimizer and loss function
lr_decay = 0.9995
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

compute_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)


# --------------------------------------------------------------------------------------------------------------
@tf.function
def train_step(batch_x, batch_y):
    with tf.GradientTape() as ae_tape:
        logits = pixelcnn(batch_x, training=True)

        logits = tf.reshape(logits, [-1, height, width, q_levels, n_channel])  # shape [N,H,W,DC] -> [N,H,W,D,C]
        logits = tf.transpose(logits, perm=[0, 1, 2, 4, 3])  # shape [N,H,W,D,C] -> [N,H,W,C,D]

        loss = compute_loss(tf.one_hot(batch_y, q_levels), logits)

    gradients = ae_tape.gradient(loss, pixelcnn.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, pixelcnn.trainable_variables))

    return loss

# --------------------------------------------------------------------------------------------------------------
# Training loop
n_epochs = 30
n_iter = int(np.ceil(x_train_quantised.shape[0] / batch_size))
for epoch in range(n_epochs):
    start_epoch = time.time()
    for i_iter, (batch_x, batch_y) in enumerate(train_dataset):
        start = time.time()
        optimizer.lr = optimizer.lr * lr_decay
        loss = train_step(batch_x, batch_y)
        iter_time = time.time() - start
        if i_iter % 100 == 0:
            print('EPOCH {:3d}: ITER {:4d}/{:4d} TIME: {:.2f} LOSS: {:.4f}'.format(epoch,
                                                                                   i_iter, n_iter,
                                                                                   iter_time,
                                                                                   loss))
    epoch_time = time.time() - start_epoch
    print('EPOCH {:3d}: TIME: {:.2f} ETA: {:.2f}'.format(epoch,
                                                         epoch_time,
                                                         epoch_time * (n_epochs - epoch)))

# --------------------------------------------------------------------------------------------------------------
# Test
test_loss = []
for batch_x, batch_y in test_dataset:
    logits = pixelcnn(batch_x, training=False)
    logits = tf.reshape(logits, [-1, height, width, q_levels, n_channel])
    logits = tf.transpose(logits, perm=[0, 1, 2, 4, 3])

    # Calculate cross-entropy (= negative log-likelihood)
    loss = compute_loss(tf.one_hot(batch_y, q_levels), logits)

    test_loss.append(loss)
print('nll : {:} nats'.format(np.array(test_loss).mean()))
print('bits/dim : {:}'.format(np.array(test_loss).mean() / (height * width)))


# --------------------------------------------------------------------------------------------------------------
# Generating new images
samples = (np.random.rand(100, height, width, n_channel) * 0.01).astype('float32')
for i in range(28):
    for j in range(28):
        logits = pixelcnn(samples)
        logits = tf.reshape(logits, [-1, height, width, q_levels, n_channel])
        logits = tf.transpose(logits, perm=[0, 1, 2, 4, 3])
        probs = tf.nn.softmax(logits)
        next_sample = probs[:, i, j, 0, :]
        samples[:, i, j, 0] = sample_from(next_sample.numpy()) / (q_levels - 1)

fig = plt.figure(figsize=(10, 10))
for x in range(1, 10):
    for y in range(1, 10):
        ax = fig.add_subplot(10, 10, 10 * y + x)
        ax.matshow(samples[10 * y + x, :, :, 0], cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()


# --------------------------------------------------------------------------------------------------------------
# Filling occluded images
occlude_start_row = 14
num_generated_images = 100
samples = np.copy(x_test_quantised[0:num_generated_images, :, :, :])
samples = samples / (q_levels - 1)
samples[:, occlude_start_row:, :, :] = 0

for i in range(occlude_start_row, height):
    for j in range(width):
        logits = pixelcnn(samples)
        logits = tf.reshape(logits, [-1, height, width, q_levels, n_channel])
        logits = tf.transpose(logits, perm=[0, 1, 2, 4, 3])
        probs = tf.nn.softmax(logits)
        next_sample = probs[:, i, j, 0, :]
        samples[:, i, j, 0] = sample_from(next_sample.numpy()) / (q_levels - 1)

fig = plt.figure(figsize=(10, 10))
for x in range(1, 10):
    for y in range(1, 10):
        ax = fig.add_subplot(10, 10, 10 * y + x)
        ax.matshow(samples[10 * y + x, :, :, 0], cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
plt.show()


# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------

class CroppedConvolution(L.Convolution2D):
def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

def __call__(self, x):
    ret = super().__call__(x)
    kh, kw = self.ksize
    pad_h, pad_w = self.pad
    h_crop = -(kh + 1) if pad_h == kh else None
    w_crop = -(kw + 1) if pad_w == kw else None

    return ret[:, :, :h_crop, :w_crop]


def down_shift(x):
xs = int_shape(x)
return tf.concat([tf.zeros([xs[0], 1, xs[2], xs[3]]), x[:, :xs[1] - 1, :, :]], 1)


def right_shift(x):
xs = int_shape(x)
return tf.concat([tf.zeros([xs[0], xs[1], 1, xs[3]]), x[:, :, :xs[2] - 1, :]], 2)



@add_arg_scope
def down_shifted_conv2d(x, num_filters, filter_size=[2, 3], stride=[1, 1], **kwargs):
x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0], [int((filter_size[1] - 1) / 2), int((filter_size[1] - 1) / 2)],
               [0, 0]])
return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)


@add_arg_scope
def down_shifted_deconv2d(x, num_filters, filter_size=[2, 3], stride=[1, 1], **kwargs):
x = deconv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)
xs = int_shape(x)
return x[:, :(xs[1] - filter_size[0] + 1), int((filter_size[1] - 1) / 2):(xs[2] - int((filter_size[1] - 1) / 2)), :]


@add_arg_scope
def down_right_shifted_conv2d(x, num_filters, filter_size=[2, 2], stride=[1, 1], **kwargs):
x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0], [filter_size[1] - 1, 0], [0, 0]])
return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)


@add_arg_scope
def down_right_shifted_deconv2d(x, num_filters, filter_size=[2, 2], stride=[1, 1], **kwargs):
x = deconv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)
xs = int_shape(x)


return x[:, :(xs[1] - filter_size[0] + 1):, :(xs[2] - filter_size[1] + 1), :]


# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# https://github.com/pclucas14/pixel-cnn-pp/blob/master/model.py
# PIXELCNN++

self.u_init = down_shifted_conv2d(input_channels + 1, nr_filters, filter_size=(2, 3),
                                  shift_output_down=True)

self.ul_init = nn.ModuleList([down_shifted_conv2d(input_channels + 1, nr_filters,
                                                  filter_size=(1, 3), shift_output_down=True),
                              down_right_shifted_conv2d(input_channels + 1, nr_filters,
                                                        filter_size=(2, 1), shift_output_right=True)])

u_list = [self.u_init(x)]
ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]

self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
self.down_shift_pad = nn.ZeroPad2d((0, 0, 1, 0))


''' utilities for shifting the image around, efficient alternative to masking convolutions '''
def down_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when downshifting, the last row is removed
    x = x[:, :, :xs[2] - 1, :]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((0, 0, 1, 0)) if pad is None else pad
    return pad(x)


def right_shift(x, pad=None):
    # Pytorch ordering
    xs = [int(y) for y in x.size()]
    # when righshifting, the last column is removed
    x = x[:, :, :, :xs[3] - 1]
    # padding left, padding right, padding top, padding bottom
    pad = nn.ZeroPad2d((1, 0, 0, 0)) if pad is None else pad
    return pad(x)



class down_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 3), stride=(1, 1),
                 shift_output_down=False, norm='weight_norm'):
        super(down_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride)
        self.shift_output_down = shift_output_down
        self.norm = norm
        self.pad = nn.ZeroPad2d((int((filter_size[1] - 1) / 2),  # pad left
                                 int((filter_size[1] - 1) / 2),  # pad right
                                 filter_size[0] - 1,  # pad top
                                 0))  # pad down

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_down:
            self.down_shift = lambda x: down_shift(x, pad=nn.ZeroPad2d((0, 0, 1, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.down_shift(x) if self.shift_output_down else x


class down_right_shifted_conv2d(nn.Module):
    def __init__(self, num_filters_in, num_filters_out, filter_size=(2, 2), stride=(1, 1),
                 shift_output_right=False, norm='weight_norm'):
        super(down_right_shifted_conv2d, self).__init__()

        assert norm in [None, 'batch_norm', 'weight_norm']
        self.pad = nn.ZeroPad2d((filter_size[1] - 1, 0, filter_size[0] - 1, 0))
        self.conv = nn.Conv2d(num_filters_in, num_filters_out, filter_size, stride=stride)
        self.shift_output_right = shift_output_right
        self.norm = norm

        if norm == 'weight_norm':
            self.conv = wn(self.conv)
        elif norm == 'batch_norm':
            self.bn = nn.BatchNorm2d(num_filters_out)

        if shift_output_right:
            self.right_shift = lambda x: right_shift(x, pad=nn.ZeroPad2d((1, 0, 0, 0)))

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.bn(x) if self.norm == 'batch_norm' else x
        return self.right_shift(x) if self.shift_output_right else x

# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# https://github.com/jonathanventura/pixelcnn/blob/master/gated_pixelcnn.py

def _apply_activation(x):
    # split in half along channel axis
    x0, x1 = tf.split(x, 2, axis=3)

    # apply separate activations
    x_tanh = tf.nn.tanh(x0)
    x_sigmoid = tf.nn.sigmoid(x1)

    # combine activations
    return x_tanh * x_sigmoid


def _conv(inputs, num_outputs, filter_size, name):
    # relu activation
    # x = tf.nn.relu(inputs)
    x = inputs

    # get number of input channels
    num_channels_in = inputs.get_shape().as_list()[-1]

    # create filter weights
    weights = tf.get_variable(name + '_weights',
                              shape=filter_size + [num_channels_in, num_outputs],
                              initializer=tf.glorot_uniform_initializer())

    # create filter bias
    bias = tf.get_variable(name + '_bias',
                           shape=(num_outputs,),
                           initializer=tf.zeros_initializer())

    # apply filter and bias
    x = tf.nn.conv2d(x, weights, [1, 1, 1, 1], 'VALID')
    x = tf.nn.bias_add(x, bias)

    return x



def _gated_pixel_cnn_layer(vinput, hinput, filter_size, num_filters, layer_index):
    """Gated activation PixelCNN layer
       Paper reference: https://arxiv.org/pdf/1606.05328.pdf
       Code reference: https://github.com/dritchie/pixelCNN/blob/master/pixelCNN.lua
    """
    k = filter_size
    floork = int(floor(filter_size / 2))
    ceilk = int(ceil(filter_size / 2))

    # kxk convolution for vertical stack
    vinput_padded = tf.pad(vinput, [[0, 0], [ceilk, 0], [floork, floork], [0, 0]])
    vconv = _conv(vinput_padded, 2 * num_filters, [ceilk, k], 'vconv_%d' % layer_index)
    vconv = vconv[:, :-1, :, :]

    # kx1 convolution for horizontal stack
    hinput_padded = tf.pad(hinput, [[0, 0], [0, 0], [ceilk, 0], [0, 0]])
    hconv = _conv(hinput_padded, 2 * num_filters, [1, ceilk], 'hconv_%d' % layer_index)
    if layer_index == 0:
        hconv = hconv[:, :, :-1, :]
    else:
        hconv = hconv[:, :, 1:, :]

    # 1x1 transitional convolution for vstack
    vconv1 = _conv(vconv, 2 * num_filters, [1, 1], 'vconv1_%d' % layer_index)

    # add vstack to hstack
    hconv = hconv + vconv1

    # apply activations
    vconv = _apply_activation(vconv)
    hconv = _apply_activation(hconv)

    # residual connection in hstack
    if layer_index > 0:
        hconv1 = _conv(hconv, num_filters, [1, 1], 'hconv1_%d' % layer_index)
        hconv = hinput + hconv1

    return vconv, hconv


def gated_pixelcnn(inputs, num_filters, num_layers, num_outputs):
    """Builds Gated PixelCNN graph.
    Args:
        inputs: input tensor (B,H,W,C)
    Returns:
        Predicted tensor
    """
    with tf.variable_scope('pixel_cnn') as sc:
        vstack = inputs
        hstack = inputs

        # first layer: masked 7x7
        vstack, hstack = _gated_pixel_cnn_layer(vstack, hstack, 7, num_filters, 0)

        # next layers: masked 3x3
        for i in range(num_layers):
            vstack, hstack = _gated_pixel_cnn_layer(vstack, hstack, 3, num_filters, i + 1)

        # final layers
        x = _conv(tf.nn.relu(hstack), num_filters, [1, 1], 'conv1')
        logits = _conv(tf.nn.relu(x), num_outputs, [1, 1], 'logits')

    return logits


