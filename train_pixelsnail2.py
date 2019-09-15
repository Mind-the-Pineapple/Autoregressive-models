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
import matplotlib



class AttentionBlock(nn.Module):
    def __init__(self, in_channels, key_size, value_size):
        super(AttentionBlock, self).__init__()
        self.linear_query = nn.Linear(in_channels, key_size)
        self.linear_keys = nn.Linear(in_channels, key_size)
        self.linear_values = nn.Linear(in_channels, value_size)
        self.sqrt_key_size = math.sqrt(key_size)

    def forward(self, input):
        # input is dim (N, T, in_channels) where N is the batch_size, and T is
        # the sequence length
        mask = np.array([[1 if i>j else 0 for i in range(input.shape[1])] for j in range(input.shape[1])])
        mask = torch.ByteTensor(mask).cuda()

        #import pdb; pdb.set_trace()
        keys = self.linear_keys(input) # shape: (N, T, key_size)
        query = self.linear_query(input) # shape: (N, T, key_size)
        values = self.linear_values(input) # shape: (N, T, value_size)
        temp = torch.bmm(query, torch.transpose(keys, 1, 2)) # shape: (N, T, T)
        temp.data.masked_fill_(mask, -float('inf'))
        temp = F.softmax(temp / self.sqrt_key_size, dim=1) # shape: (N, T, T), broadcasting over any slice [:, x, :], each row of the matrix
        temp = torch.bmm(temp, values) # shape: (N, T, value_size)
        return torch.cat((input, temp), dim=2) # shape: (N, T, in_channels + value_size)


class CausalAttention(tf.keras.Model):
    def __init__(self, key_size=16, value_size=128):
        super(CausalAttention, self).__init__()

        self.linear_query = tf.keras.layers.Dense(key_size, activation='linear')
        self.linear_keys = tf.keras.layers.Dense(key_size, activation='linear')
        self.linear_values = tf.keras.layers.Dense(value_size, activation='linear')
        self.sqrt_key_size = tf.sqrt(key_size)

    def call(self, input_tensor):
        # input_tensor is dim (N, T, in_channels) where N is the batch_size, and T is

        batch_size, height, width, _ = input_tensor.shape


        keys = self.linear_keys(input_tensor) # shape: (N, T, key_size)
        query = self.linear_query(input_tensor) # shape: (N, T, key_size)
        values = self.linear_values(input_tensor) # shape: (N, T, value_size)

        attn = tf.matmul(query, keys) / self.sqrt_key_size
        mask, start_mask = self._causal_mask(height * width)
        # attn = attn.masked_fill(mask == 0, -1e4)
        attn = tf.nn.softmax(attn, axis=3) * start_mask # TODO: Verificar axis
        out = tf.matmul(attn, values) # OK
        out = tf.transpose(out, perm=[1,2])
        out = tf.reshape(out, [batch_size, height, width, self.dim_head * self.n_head]) # TODO: checar dim_head e n_head

        return out



class CausalAttention(tf.keras.layers.Layer):
    def __init__(self,
                 query_channel,
                 key_channel,
                 value_size,
                 channel,
                 n_head=8,
                 kernel_initializer='glorot_uniform'):
        super(CausalAttention, self).__init__()

        self.query_channel = query_channel
        self.key_channel = key_channel
        self.channel = channel
        self.dim_head = channel // n_head
        self.n_head = n_head

        self.query = self.add_variable("query",
                                        shape=(self.query_channel,
                                               self.channel),
                                        initializer=self.kernel_initializer,
                                        trainable=True)
        self.key = self.add_variable("key",
                                        shape=(self.key_channel,
                                               self.channel),
                                        initializer=self.kernel_initializer,
                                        trainable=True)
        self.value = self.add_variable("value",
                                        shape=(self.key_channel,
                                               self.channel),
                                        initializer=self.kernel_initializer,
                                        trainable=True)

        self.kernel_initializer = keras.initializers.get(kernel_initializer)

    def _causal_mask(size):
        shape = [size, size]
        mask = np.triu(np.ones(shape), k=1).astype(np.uint8).T
        start_mask = np.ones(size).astype(np.float32)
        start_mask[0] = 0

        return (
            tf.expand_dims(mask, axis=0),
            tf.expand_dims(start_mask, axis=1),
        )

    def masked_fill(matrix, mask, num=-1e4):
        # TODO:
        negmask = 1 - mask
        return (matrix * mask) + (-((negmask * num + num) - num)

    def forward(self, query, key):
        batch, _, height, width = key.shape

        def reshape(input):
            return input.view(batch, -1, self.n_head, self.dim_head).transpose(1, 2)

        query_flat = query.view(batch, query.shape[1], -1).transpose(1, 2)
        key_flat = key.view(batch, key.shape[1], -1).transpose(1, 2)
        query = reshape(self.query(query_flat))
        key = reshape(self.key(key_flat)).transpose(2, 3)
        value = reshape(self.value(key_flat))

        attn = tf.matmul(query, key) / tf.sqrt(self.dim_head) # OK
        mask, start_mask = self._causal_mask(height * width)
        # attn = attn.masked_fill(mask == 0, -1e4)
        attn = tf.nn.softmax(attn, axis=3) * start_mask # TODO: Verificar axis
        out = tf.matmul(attn, value) # OK
        out = tf.transpose(out, perm=[1,2])
        out = tf.reshape(batch, height, width, self.dim_head * self.n_head) # TODO: checar dim_head e n_head

        return out






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






class MaskedConv2D(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 mask_type='B',
                 strides=1,
                 padding='same',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        super(MaskedConv2D, self).__init__()

        assert mask_type in {'A', 'B', 'V'}
        self.mask_type = mask_type

        self.strides = strides
        self.filters = filters

        if isinstance(kernel_size, int):
            kernel_size =  (kernel_size,) * 2
        self.kernel_size = kernel_size

        self.padding = padding.upper()
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        kernel_h, kernel_w = self.kernel_size

        self.kernel = self.add_variable("kernel",
                                        shape=(kernel_h,
                                               kernel_w,
                                               int(input_shape[-1]),
                                               self.filters),
                                        initializer=self.kernel_initializer,
                                        trainable=True)

        self.bias = self.add_variable("bias",
                                      shape=(self.filters, ),
                                      initializer=self.bias_initializer,
                                      trainable=True)

        mask = np.ones(self.kernel.shape, dtype=np.float32)
        if self.mask_type == 'V':
            mask[kernel_h // 2:, :, :, :] = 0.
        else:
            mask[kernel_h // 2, kernel_w // 2 + (self.mask_type == 'B'):, :, :] = 0.
            mask[kernel_h // 2 + 1:, :, :] = 0.

        self.mask = tf.constant(mask,
                                dtype=tf.float32,
                                name='mask')

    def call(self, input):
        masked_kernel = tf.math.multiply(self.mask, self.kernel)
        x = tf.nn.conv2d(input, masked_kernel, strides=[1, self.strides, self.strides, 1], padding=self.padding)
        x = tf.nn.bias_add(x, self.bias)
        return x



class GatedBlock(tf.keras.Model):
    def __init__(self, n_filters, kernel_size):
        super(GatedBlock, self).__init__(name='')

        self.vertical_conv = MaskedConv2D(2 * n_filters, kernel_size=(kernel_size, kernel_size), mask_type='V')
        self.horizontal_conv = MaskedConv2D(2 * n_filters, kernel_size=(1, kernel_size))
        self.v_to_h_conv = keras.layers.Conv2D(filters=2 * n_filters, kernel_size=1)

        self.horizontal_output = keras.layers.Conv2D(filters=n_filters, kernel_size=1)

    def _gate(self, x):
        tanh_preactivation, sigmoid_preactivation =  tf.split(x, 2, axis=-1)
        return tf.nn.tanh(tanh_preactivation) * tf.nn.sigmoid(sigmoid_preactivation)

    def call(self, input_tensor):
        v, h = tf.split(input_tensor, 2, axis=-1)
        horizontal_preactivation = self.horizontal_conv(h)  # 1xN
        vertical_preactivation = self.vertical_conv(v)  # NxN
        v_to_h = self.v_to_h_conv(vertical_preactivation)  # 1x1
        v_out = self._gate(vertical_preactivation)

        horizontal_preactivation = horizontal_preactivation + v_to_h
        h_activated = self._gate(horizontal_preactivation)

        h_preres = self.horizontal_output(h_activated)

        h_out = h + h_preres

        output = tf.concat((v_out, h_out), axis=-1)
        return output



def quantisize(images, levels):
    return (np.digitize(images, np.arange(levels) / levels) - 1).astype('int32')


def binarize(images):
    """
    Stochastically binarize values in [0, 1] by treating them as p-values of
    a Bernoulli distribution.
    """
    return (np.random.uniform(size=images.shape) < images).astype('float32')

def sample_from(distribution):
    batch_size, bins = distribution.shape
    return np.array([np.random.choice(bins, p=distr) for distr in distribution])

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

    batch_size = 100
    train_buf = 60000

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_quantised.astype('float32')/(q_levels-1), x_train_quantised))
    train_dataset = train_dataset.shuffle(buffer_size=train_buf)
    train_dataset = train_dataset.batch(batch_size)

    # https://github.com/RishabGoel/PixelCNN/blob/master/pixel_cnn.py
    # https://github.com/jonathanventura/pixelcnn/blob/master/pixelcnn.py
    n_channel = 1

    inputs = keras.layers.Input(shape=(28, 28, 1))
    x = MaskedConv2D(mask_type='A', filters=128, kernel_size=7, strides=1)(inputs)
    for i in range(7):
        x = GatedBlock(n_filters=64, kernel_size=3)(x)
    v, h = tf.split(x, 2, axis=-1)
    x = keras.layers.Activation(activation='relu')(h)
    x = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1)(x)
    x = keras.layers.Activation(activation='relu')(x)
    x = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1)(x)
    x = keras.layers.Conv2D(filters=n_channel * q_levels, kernel_size=1, strides=1)(x)  # shape [N,H,W,DC]

    pixelcnn = tf.keras.Model(inputs=inputs, outputs=x)


    learning_rate = 3e-4
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    compute_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_step(batch_x, batch_y):
        with tf.GradientTape() as ae_tape:
            logits = pixelcnn(batch_x)

            logits = tf.reshape(logits, [-1, 28, 28, q_levels, n_channel])  # shape [N,H,W,DC] -> [N,H,W,D,C]
            logits = tf.transpose(logits, perm=[0, 1, 2, 4, 3])  # shape [N,H,W,D,C] -> [N,H,W,C,D]

            loss  = compute_loss(tf.one_hot(batch_y, q_levels) , logits)

        gradients = ae_tape.gradient(loss, pixelcnn.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        optimizer.apply_gradients(zip(gradients, pixelcnn.trainable_variables))

        return loss

    background = tf.concat(
        [
            ((tf.range(28, dtype=tf.float32) - 28 / 2) / 28)[None, :, None, None] + 0. * x,
            ((tf.range(28, dtype=tf.float32) - 28 / 2) / 28)[None, None, :, None] + 0. * x,
        ],
        axis=3
    )


    epochs = 5
    for epoch in range(epochs):
        print(epoch)
        for batch_x, batch_y in train_dataset:
            print()
            loss = train_step(batch_x, batch_y)
            print(loss)


    samples = (np.random.rand(100, 28, 28, 1)* 0.01).astype('float32')
    for i in range(28):
        for j in range(28):
            A = pixelcnn(samples)
            A = tf.reshape(A, [-1, 28, 28, q_levels, n_channel])  # shape [N,H,W,DC] -> [N,H,W,D,C]
            A = tf.transpose(A, perm=[0, 1, 2, 4, 3])  # shape [N,H,W,D,C] -> [N,H,W,C,D]
            B = tf.nn.softmax(A)
            next_sample = B[:,i,j,0,:]
            samples[:, i, j, 0] = sample_from(next_sample.numpy()) / (q_levels - 1)
            print("{} {}: {}".format(i, j, sample_from(next_sample.numpy())[0]))

    fig = plt.figure()
    for x in range(1,10):
        for y in range(1, 10):
            ax = fig.add_subplot(10, 10, 10 * y + x)
            ax.matshow(samples[10 * y + x,:,:,0], cmap=matplotlib.cm.binary)
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
    return tf.concat([tf.zeros([xs[0],1,xs[2],xs[3]]), x[:,:xs[1]-1,:,:]],1)



def right_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],xs[1],1,xs[3]]), x[:,:,:xs[2]-1,:]],2)

class CroppedConvolution(tf.keras.layers.Layer):

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros'):
        super(CroppedConvolution, self).__init__()

        self.strides = strides
        self.filters = filters
        self.kernel_size = kernel_size
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

        h_crop = -(kh + 1) if pad_h == kh else None
        w_crop = -(kw + 1) if pad_w == kw else None
        pad = [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]

    def call(self, input):

        x = tf.nn.conv2d(input, self.kernel, strides=[1, self.strides, self.strides, 1], padding=self.padding)
        x = tf.nn.bias_add(x, self.bias)
        return x

@add_arg_scope
def down_shifted_conv2d(x, num_filters, filter_size=[2,3], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1,0], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)

@add_arg_scope
def down_shifted_deconv2d(x, num_filters, filter_size=[2,3], stride=[1,1], **kwargs):
    x = deconv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)
    xs = int_shape(x)
    return x[:,:(xs[1]-filter_size[0]+1),int((filter_size[1]-1)/2):(xs[2]-int((filter_size[1]-1)/2)),:]

@add_arg_scope
def down_right_shifted_conv2d(x, num_filters, filter_size=[2,2], stride=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1, 0], [filter_size[1]-1, 0],[0,0]])
    return conv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)

@add_arg_scope
def down_right_shifted_deconv2d(x, num_filters, filter_size=[2,2], stride=[1,1], **kwargs):
    x = deconv2d(x, num_filters, filter_size=filter_size, pad='VALID', stride=stride, **kwargs)
    xs = int_shape(x)
return x[:,:(xs[1]-filter_size[0]+1):,:(xs[2]-filter_size[1]+1),:]
