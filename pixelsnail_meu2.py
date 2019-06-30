import tensorflow as tf
from tensorflow import keras

def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape()) - 1
    return tf.nn.elu(tf.concat([x, -x], axis))


@add_arg_scope
def nin(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = dense(x, num_units, **kwargs)
    return tf.reshape(x, s[:-1] + [num_units])



@add_arg_scope
def gated_resnet(x, a=None, h=None, nonlinearity=concat_elu, conv=conv2d, init=False, counters={}, ema=None, dropout_p=0., **kwargs):
    xs = int_shape(x)
    num_filters = xs[-1]

    c1 = keras.layers.Conv2D(filters=num_filters, kernel_size=(3,3))
    if a is not None:
        c1 += nin(nonlinearity(a), num_filters)
    c1 = nonlinearity(c1)
    c1 = keras.layers.Dropout(dropout_p)(c1)
    c2 = conv(c1, num_filters * 2, init_scale=0.1)


    c1 = conv(nonlinearity(x), num_filters)
    if a is not None:  # add short-cut connection if auxiliary input 'a' is given
        c1 += nin(nonlinearity(a), num_filters)
    c1 = nonlinearity(c1)
    if dropout_p > 0:
        c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
    c2 = conv(c1, num_filters * 2, init_scale=0.1)

    # add projection of h vector if included: conditional generation
    if h is not None:
        with tf.variable_scope(get_name('conditional_weights', counters)):
            hw = get_var_maybe_avg('hw', ema, shape=[int_shape(h)[-1], 2 * num_filters], dtype=tf.float32,
                                   initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        if init:
            hw = hw.initialized_value()
        c2 += tf.reshape(tf.matmul(h, hw), [xs[0], 1, 1, 2 * num_filters])

    # Is this 3,2 or 2,3 ?
    a, b = tf.split(c2, 2, 3)
    c3 = a * tf.nn.sigmoid(b)
    return x + c3