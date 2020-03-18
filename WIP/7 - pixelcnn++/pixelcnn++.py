"""

https://github.com/openai/pixel-cnn/blob/master/pixel_cnn_pp/nn.py
https://github.com/Rayhane-mamah/Tacotron-2/issues/155
https://github.com/r9y9/wavenet_vocoder/blob/master/wavenet_vocoder/mixture.py
https://github.com/bjlkeng/sandbox/blob/master/notebooks/pixel_cnn/pixelcnn.ipynb

Fazer o toy exemplo de
http://bjlkeng.github.io/posts/pixelcnn/

"""
import random as rn
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keepdims=True)
    return m + tf.math.log(tf.reduce_sum(tf.exp(x - m2), axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape()) - 1
    m = tf.reduce_max(x, axis, keepdims=True)
    return x - m - tf.math.log(tf.reduce_sum(tf.exp(x - m), axis, keepdims=True))




def discretized_mix_logistic_loss(x, l, sum_all=True):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    xs = x.shape  # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    ls = l.shape  # predicted distribution, e.g. (B,32,32,100)

    # ------------------------------------------------------------------------------------
    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])

    means = l[:, :, :, :, :nr_mix]
    log_scales = tf.maximum(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    coeffs = tf.nn.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix])

    # ------------------------------------------------------------------------------------
    # here and below: getting the means and adjusting them based on preceding sub-pixels
    x = tf.reshape(x, xs + [1]) + tf.zeros(xs + [nr_mix])
    m2 = tf.reshape(means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix])
    m3 = tf.reshape(
        means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :],
        [xs[0], xs[1], xs[2], 1, nr_mix])

    means = tf.concat([tf.reshape(means[:, :, :, 0, :], [xs[0], xs[1], xs[2], 1, nr_mix]), m2, m3], 3)
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)

    # ------------------------------------------------------------------------------------
    # log probability for edge case of 0 (before scaling)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in)

    # ------------------------------------------------------------------------------------
    # log probability for edge case of 255 (before scaling)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = tf.nn.sigmoid(min_in)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in)

    # ------------------------------------------------------------------------------------
    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2. * tf.nn.softplus( mid_in)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.where(x < -0.999,
                         log_cdf_plus,
                         tf.where(x > 0.999,
                                  log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.math.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))

    log_probs = tf.reduce_sum(log_probs, 3) + log_prob_from_logits(logit_probs)

    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        return -tf.reduce_sum(log_sum_exp(log_probs), [1, 2])


def sample_from_discretized_mix_logistic(l, nr_mix):
    ls = l.shape
    xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix]
    l = tf.reshape(l[:, :, :, nr_mix:], xs + [nr_mix * 3])

    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(
        logit_probs - tf.math.log(-tf.math.log(tf.random.uniform(logit_probs.get_shape(), minval=1e-5, maxval=1. - 1e-5))), 3),
                     depth=nr_mix, dtype=tf.float32)
    sel = tf.reshape(sel, xs[:-1] + [1, nr_mix])

    # select logistic parameters
    means = tf.reduce_sum(l[:, :, :, :, :nr_mix] * sel, 4)
    log_scales = tf.maximum(tf.reduce_sum(l[:, :, :, :, nr_mix:2 * nr_mix] * sel, 4), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, 4)

    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random.uniform(means.get_shape(), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales) * (tf.math.log(u) - tf.math.log(1. - u))

    x0 = tf.minimum(tf.maximum(x[:, :, :, 0], -1.), 1.)
    x1 = tf.minimum(tf.maximum(x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, -1.), 1.)
    x2 = tf.minimum(tf.maximum(x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, -1.), 1.)

    return tf.concat([tf.reshape(x0, xs[:-1] + [1]), tf.reshape(x1, xs[:-1] + [1]), tf.reshape(x2, xs[:-1] + [1])], 3)

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape())-1
    return tf.nn.elu(tf.concat([x, -x], axis))

def nin(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    s = x.shape
    x = tf.reshape(x, [np.prod(s[:-1]),s[-1]])
    x = keras.layers.Dense(num_units, **kwargs)(x)
    return tf.reshape(x, s[:-1]+[num_units])

def gated_resnet(x, a=None, conv=keras.layers.Conv2D, dropout_p=0., **kwargs):
    xs = x.shape
    num_filters = xs[-1]

    c1 = conv(concat_elu(x), num_filters)
    if a is not None: # add short-cut connection if auxiliary input 'a' is given
        c1 += nin(concat_elu(a), num_filters)
    c1 = concat_elu(c1)
    if dropout_p > 0:
        c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
    c2 = conv(c1, num_filters * 2)
    a, b = tf.split(c2, 2, 3)
    c3 = a * tf.nn.sigmoid(b)
    return x + c3

''' utilities for shifting the image around, efficient alternative to masking convolutions '''

def down_shift(x):
    xs = x.shape
    return tf.concat([tf.zeros([xs[0],1,xs[2],xs[3]]), x[:,:xs[1]-1,:,:]],1)

def right_shift(x):
    xs = x.shape
    return tf.concat([tf.zeros([xs[0],xs[1],1,xs[3]]), x[:,:,:xs[2]-1,:]],2)

def down_shifted_conv2d(x, num_filters, filter_size=[2,3], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1,0], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    return keras.layers.Conv2D(num_filters, kernel_size=filter_size, padding='valid', strides=stride, **kwargs)(x)

def down_shifted_deconv2d(x, num_filters, filter_size=[2,3], stride=[1,1], **kwargs):
    x = keras.layers.Conv2DTranspose(num_filters, kernel_size=filter_size, padding='valid', strides=stride, **kwargs)(x)
    xs = x.shape
    return x[:,:(xs[1]-filter_size[0]+1),int((filter_size[1]-1)/2):(xs[2]-int((filter_size[1]-1)/2)),:]

def down_right_shifted_conv2d(x, num_filters, filter_size=[2,2], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1, 0], [filter_size[1]-1, 0],[0,0]])
    return keras.layers.Conv2D(num_filters, kernel_size=filter_size, padding='valid', strides=stride, **kwargs)(x)

def down_right_shifted_deconv2d(x, num_filters, filter_size=[2,2], stride=[1,1], **kwargs):
    x = keras.layers.Conv2DTranspose(num_filters, kernel_size=filter_size, padding='valid', strides=stride, **kwargs)(x)
    xs = x.shape
    return x[:,:(xs[1]-filter_size[0]+1):,:(xs[2]-filter_size[1]+1),:]

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

x_train = (x_train.astype('float32') / 127.5) - 1
x_test = (x_test.astype('float32') / 127.5) - 1

x_train = x_train.reshape(x_train.shape[0], height, width, 1)
x_test = x_test.reshape(x_test.shape[0], height, width, 1)


batch_size = 128
train_buf = 60000

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, x_train))
train_dataset = train_dataset.shuffle(buffer_size=train_buf)
train_dataset = train_dataset.batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, x_test))
test_dataset = test_dataset.batch(batch_size)

nr_logistic_mix = 5

dropout_p=0.5
nr_resnet=5
nr_filters=160

# ////////// up pass through pixelCNN ////////
xs = x.shape
# add channel of ones to distinguish image from padding later on
x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], 3)

# stream for pixels above
u_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))]
# stream for up and to the left
ul_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) + \
           right_shift(down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]

for rep in range(nr_resnet):
    u_list.append(gated_resnet(u_list[-1], conv=down_shifted_conv2d))
    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))

u_list.append(down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
ul_list.append(down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))

for rep in range(nr_resnet):
    u_list.append(gated_resnet(u_list[-1], conv=down_shifted_conv2d))
    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))

u_list.append(down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
ul_list.append(down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))

for rep in range(nr_resnet):
    u_list.append(gated_resnet(u_list[-1], conv=down_shifted_conv2d))
    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))

# /////// down pass ////////
u = u_list.pop()
ul = ul_list.pop()
for rep in range(nr_resnet):
    u = gated_resnet(u, u_list.pop(), conv=down_shifted_conv2d)
    ul = gated_resnet(ul, tf.concat([u, ul_list.pop()], 3), conv=down_right_shifted_conv2d)

u = down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
ul = down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])

for rep in range(nr_resnet + 1):
    u = gated_resnet(u, u_list.pop(), conv=down_shifted_conv2d)
    ul = gated_resnet(ul, tf.concat([u, ul_list.pop()], 3), conv=down_right_shifted_conv2d)

u = down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
ul = down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])

for rep in range(nr_resnet + 1):
    u = gated_resnet(u, u_list.pop(), conv=down_shifted_conv2d)
    ul = gated_resnet(ul, tf.concat([u, ul_list.pop()], 3), conv=down_right_shifted_conv2d)
