
OK gated_resnet
OK nin
OK conv2d
OK deconv2d
OK down_shift
OK down_shifted_conv2d
OK concat_elu
OK right_shift
OK down_right_shifted_conv2d
OK causal_attention
OK dense





@add_arg_scope
def down_right_shifted_conv2d(x, num_filters, filter_size=[2, 2], stride=[1, 1], **kwargs):
    x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0],
                   [filter_size[1] - 1, 0], [0, 0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)



@add_arg_scope
def down_shifted_conv2d(x, num_filters, filter_size=[2, 3], stride=[1, 1], **kwargs):
    x = tf.pad(x, [[0, 0], [filter_size[0] - 1, 0],
                   [int((filter_size[1] - 1) / 2), int((filter_size[1] - 1) / 2)], [0, 0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)


def int_shape(x):
    return list(map(int, x.get_shape()))

def down_shift(x, step=1):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0], step, xs[2], xs[3]]), x[:, :xs[1] - step, :, :]], 1)


def right_shift(x, step=1):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0], xs[1], step, xs[3]]), x[:, :, :xs[2] - step, :]], 2)



@add_arg_scope
def gated_resnet(x, a=None, h=None, nonlinearity=concat_elu, conv=conv2d, init=False, counters={}, ema=None, dropout_p=0., **kwargs):
    xs = int_shape(x)
    num_filters = xs[-1]

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



@add_arg_scope
def nin(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = dense(x, num_units, **kwargs)
    return tf.reshape(x, s[:-1] + [num_units])

''' meta-layer consisting of multiple base layers '''




@add_arg_scope
def deconv2d(x, num_filters, filter_size=[3, 3], stride=[1, 1], pad='SAME', nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
    ''' transposed convolutional layer '''
    name = get_name('deconv2d', counters)
    xs = int_shape(x)
    if pad == 'SAME':
        target_shape = [xs[0], xs[1] * stride[0],
                        xs[2] * stride[1], num_filters]
    else:
        target_shape = [xs[0], xs[1] * stride[0] + filter_size[0] -
                        1, xs[2] * stride[1] + filter_size[1] - 1, num_filters]
    with tf.variable_scope(name):
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V', filter_size + [num_filters, int(x.get_shape(
            )[-1])], tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 3])
            x_init = tf.nn.conv2d_transpose(x, V_norm, target_shape, [
                                            1] + stride + [1], padding=pad)
            m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=tf.float32,
                                initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32,
                                initializer=-m_init * scale_init, trainable=True)
            x_init = tf.reshape(scale_init, [
                                1, 1, 1, num_filters]) * (x_init - tf.reshape(m_init, [1, 1, 1, num_filters]))
            if nonlinearity is not None:
                x_init = nonlinearity(x_init)
            return x_init

        else:
            V, g, b = get_vars_maybe_avg(['V', 'g', 'b'], ema)
            # tf.assert_variables_initialized([V, g, b])

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, num_filters, 1]) * \
                tf.nn.l2_normalize(V, [0, 1, 3])

            # calculate convolutional layer output
            x = tf.nn.conv2d_transpose(
                x, W, target_shape, [1] + stride + [1], padding=pad)
            x = tf.nn.bias_add(x, b)
            # apply nonlinearity
            if nonlinearity is not None:
                x = nonlinearity(x)
            return x





@add_arg_scope
def conv2d(x, num_filters, filter_size=[3, 3], stride=[1, 1], pad='SAME', nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
    ''' convolutional layer '''
    name = get_name('conv2d', counters)
    with tf.variable_scope(name):
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V', filter_size + [int(x.get_shape()[-1]), num_filters],
                                tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1, 2])
            x_init = tf.nn.conv2d(x, V_norm, [1] + stride + [1], pad)
            m_init, v_init = tf.nn.moments(x_init, [0, 1, 2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=tf.float32,
                                initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32,
                                initializer=-m_init * scale_init, trainable=True)
            x_init = tf.reshape(scale_init, [
                                1, 1, 1, num_filters]) * (x_init - tf.reshape(m_init, [1, 1, 1, num_filters]))
            if nonlinearity is not None:
                x_init = nonlinearity(x_init)
            return x_init

        else:
            V, g, b = get_vars_maybe_avg(['V', 'g', 'b'], ema)
            # tf.assert_variables_initialized([V, g, b])

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, 1, num_filters]) * \
                tf.nn.l2_normalize(V, [0, 1, 2])

            # calculate convolutional layer output
            x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride + [1], pad), b)

            # apply nonlinearity
            if nonlinearity is not None:
                x = nonlinearity(x)
return x





@add_arg_scope
def dense(x, num_units, nonlinearity=None, init_scale=1., counters={}, init=False, ema=None, **kwargs):
    ''' fully connected layer '''
    name = get_name('dense', counters)
    with tf.variable_scope(name):
        if init:
            # data based initialization of parameters
            V = tf.get_variable('V', [int(x.get_shape()[
                                1]), num_units], tf.float32, tf.random_normal_initializer(0, 0.05), trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0])
            x_init = tf.matmul(x, V_norm)
            m_init, v_init = tf.nn.moments(x_init, [0])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            g = tf.get_variable('g', dtype=tf.float32,
                                initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32,
                                initializer=-m_init * scale_init, trainable=True)
            x_init = tf.reshape(
                scale_init, [1, num_units]) * (x_init - tf.reshape(m_init, [1, num_units]))
            if nonlinearity is not None:
                x_init = nonlinearity(x_init)
            return x_init

        else:
            V, g, b = get_vars_maybe_avg(['V', 'g', 'b'], ema)
            # tf.assert_variables_initialized([V, g, b])

            # use weight normalization (Salimans & Kingma, 2016)
            x = tf.matmul(x, V)
            scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
            x = tf.reshape(scaler, [1, num_units]) * \
                x + tf.reshape(b, [1, num_units])

            # apply nonlinearity
            if nonlinearity is not None:
                x = nonlinearity(x)
return x



def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape()) - 1
    return tf.nn.elu(tf.concat([x, -x], axis))

def causal_attention(key, mixin, query, downsample=1, use_pos_enc=False):
    bs, nr_chns = int_shape(key)[0], int_shape(key)[-1]


    if downsample > 1:
        pool_shape = [1, downsample, downsample, 1]
        key = tf.nn.max_pool(key, pool_shape, pool_shape, 'SAME')
        mixin = tf.nn.max_pool(mixin, pool_shape, pool_shape, 'SAME')

    xs = int_shape(mixin)
    if use_pos_enc:
        pos1 = tf.range(0., xs[1]) / xs[1]
        pos2 = tf.range(0., xs[2]) / xs[1]
        mixin = tf.concat([
            mixin,
            tf.tile(pos1[None, :, None, None], [xs[0], 1, xs[2], 1]),
            tf.tile(pos2[None, None, :, None], [xs[0], xs[2], 1, 1]),
        ], axis=3)


    mixin_chns = int_shape(mixin)[-1]
    canvas_size = int(np.prod(int_shape(key)[1:-1]))
    canvas_size_q = int(np.prod(int_shape(query)[1:-1]))
    causal_mask = get_causal_mask(canvas_size_q, downsample)

    dot = tf.matmul(
        tf.reshape(query, [bs, canvas_size_q, nr_chns]),
        tf.reshape(key, [bs, canvas_size, nr_chns]),
        transpose_b=True
    ) - (1. - causal_mask) * 1e10
    dot = dot - tf.reduce_max(dot, axis=-1, keep_dims=True)

    causal_exp_dot = tf.exp(dot / np.sqrt(nr_chns).astype(np.float32)) * causal_mask
    causal_probs = causal_exp_dot / (tf.reduce_sum(causal_exp_dot, axis=-1, keep_dims=True) + 1e-6)

    mixed = tf.matmul(
        causal_probs,
        tf.reshape(mixin, [bs, canvas_size, mixin_chns])
    )

    return tf.reshape(mixed, int_shape(query)[:-1] + [mixin_chns])








def _base_noup_smallkey_spec(x, h=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=256, attn_rep=12, nr_logistic_mix=10, att_downsample=1, resnet_nonlinearity='concat_elu'):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense, nn.nin], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' +
                  resnet_nonlinearity + ' is not supported')

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, h=h):

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            background = tf.concat(
                    [
                        ((tf.range(xs[1], dtype=tf.float32) - xs[1] / 2) / xs[1])[None, :, None, None] + 0. * x,
                        ((tf.range(xs[2], dtype=tf.float32) - xs[2] / 2) / xs[2])[None, None, :, None] + 0. * x,
                    ],
                    axis=3
                    )
            # add channel of ones to distinguish image from padding later on
            x_pad = tf.concat([x, tf.ones(xs[:-1] + [1])], axis=3)

            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1, 3])) +
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 1]))]  # stream for up and to the left

            for attn_rep in range(attn_rep):
                for rep in range(nr_resnet):
                    ul_list.append(nn.gated_resnet(
                        ul_list[-1], conv=nn.down_right_shifted_conv2d))

                ul = ul_list[-1]
                raw_content = tf.concat([x, ul, background], axis=3)
                q_size = 16
                raw = nn.nin(nn.gated_resnet(raw_content, conv=nn.nin), nr_filters // 2 + q_size)
                key, mixin = raw[:, :, :, :q_size], raw[:, :, :, q_size:]
                raw_q = tf.concat([ul, background], axis=3)
                query = nn.nin(nn.gated_resnet(raw_q, conv=nn.nin), q_size)
                mixed = nn.causal_attention(key, mixin, query, downsample=att_downsample)

                ul_list.append(nn.gated_resnet(ul, mixed, conv=nn.nin))


            x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10 * nr_logistic_mix)

            return x_out



