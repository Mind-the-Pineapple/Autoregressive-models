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


    q_levels = 4
    x_train_quantised = quantisize(x_train,q_levels)
    x_test_quantised = quantisize(x_test,q_levels)

    batch_size = 100
    train_buf = 60000

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_quantised.astype('float32')/(q_levels-1), x_train_quantised))
    train_dataset = train_dataset.shuffle(buffer_size=train_buf)
    train_dataset = train_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_quantised.astype('float32')/(q_levels-1), x_test_quantised))
    test_dataset = test_dataset.batch(batch_size)


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

    test_loss = []
    for batch_x, batch_y in test_dataset:
        logits = pixelcnn(batch_x, training=False)

        logits = tf.reshape(logits, [-1, 28, 28, q_levels, n_channel])  # shape [N,H,W,DC] -> [N,H,W,D,C]
        logits = tf.transpose(logits, perm=[0, 1, 2, 4, 3])  # shape [N,H,W,D,C] -> [N,H,W,C,D]

        loss = compute_loss(tf.one_hot(batch_y, q_levels), logits)
        test_loss.append(loss)
    print('nll:{:}'.format(np.array(test_loss).mean()))
    print('bits/dim:{:}'.format(np.array(test_loss).mean()/(28*28)))


    occlude_start_row = 14
    num_generated_images = 100
    samples = np.copy(x_test_quantised[0:num_generated_images, :, :, :]).astype('float32')
    samples[:, occlude_start_row:, :, :] = 0.
    samples_labels = (np.ones((100, 1)) * 8).astype('int32')

    for i in range(occlude_start_row, 28):
        for j in range(28):
            A = pixelcnn([samples, samples_labels])
            A = tf.reshape(A, [-1, 28, 28, q_levels, n_channel])  # shape [N,H,W,DC] -> [N,H,W,D,C]
            A = tf.transpose(A, perm=[0, 1, 2, 4, 3])  # shape [N,H,W,D,C] -> [N,H,W,C,D]
            B = tf.nn.softmax(A)
            next_sample = B[:, i, j, 0, :]
            samples[:, i, j, 0] = sample_from(next_sample.numpy()) / (q_levels - 1)
            print("{} {}: {}".format(i, j, sample_from(next_sample.numpy())[0]))


# -------------------------------------------------------------------------------------------------------------
# TO CALCULATE NLL
# FROM https://github.com/Schlumberger/pixel-constrained-cnn-pytorch/blob/master/pixconcnn/models/gated_pixelcnn.py

    def log_likelihood(self, device, samples):
        """Calculates log likelihood of samples under model.
        Parameters
        ----------
        device : torch.device instance
        samples : torch.Tensor
            Batch of images. Shape (batch_size, num_channels, width, height).
            Values should be integers in [0, self.prior_net.num_colors - 1].
        """
        # Set model to evaluation mode
        self.eval()

        num_samples, num_channels, height, width = samples.size()
        log_probs = torch.zeros(num_samples)
        log_probs = log_probs.to(device)

        # Normalize samples before passing through model
        norm_samples = samples.float() / (self.num_colors - 1)
        # Calculate pixel probs according to the model
        logits = self.forward(norm_samples)
        # Note that probs has shape
        # (batch, num_colors, channels, height, width)
        probs = F.softmax(logits, dim=1)

        # Calculate probability of each pixel
        for i in range(height):
            for j in range(width):
                for k in range(num_channels):
                    # Get the batch of true values at pixel (k, i, j)
                    true_vals = samples[:, k, i, j]
                    # Get probability assigned by model to true pixel
                    probs_pixel = probs[:, true_vals, k, i, j][:, 0]
                    # Add log probs (1e-9 to avoid log(0))
                    log_probs += torch.log(probs_pixel + 1e-9)

# -------------------------------------------------------------------------------------------------------------
# From https://github.com/ShengjiaZhao/PixelCNN/blob/3075ae9e4417b1c78612ecdf328a8c49259f863f/models.py
self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fc2, labels=self.X))
self.nll = self.loss * conf.img_width * conf.img_height
self.sample_nll = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fc2, labels=self.X), axis=[1, 2, 3])

# -------------------------------------------------------------------------------------------------------------
# From https://github.com/kkleidal/GatedPixelCNNPyTorch/blob/master/mnist_pixelcnn_train.py


def evaluate(self, show_prog=False):
    nll_total = 0
    Ntotal = 0
    if show_prog:
        print("Evaluating...")
        prog = tqdm.tqdm(total=len(self.ds))
    for i, (labels, x, _) in enumerate(self.loader, 0):
        N, labels, x, x_quant = preprocess(self.args, labels, x)
        logits = self.model(x, labels=labels)
        nll = negative_log_likelihood(logits, x_quant)
        nll_total = nll_total + nll.data * N
        Ntotal = Ntotal + N
        if show_prog:
            prog.update(N)
    if show_prog:
        prog.close()
    nll = nll_total / Ntotal
    return nll


def quantisize(images, levels):
    return (np.digitize(images, np.arange(levels) / levels) - 1).astype('i')

def preprocess(args, labels, x):
    x_quant = torch.from_numpy(quantisize(x.numpy(),
        args.levels)).type(torch.LongTensor)
    x = x_quant.float() / (args.levels - 1)
    if args.gpu:
        x = x.cuda()
        x_quant = x_quant.cuda()
        labels = labels.cuda()
    x = Variable(x, requires_grad=False)
    x_quant = Variable(x_quant, requires_grad=False)
    N = x.size(0)
    labels = Variable(labels, requires_grad=False)
    return N, labels, x, x_quant


def negative_log_likelihood(logits, x_quant, dim=1):
    log_probs = F.log_softmax(logits, dim=dim)
    flatten = lambda x, shape: x.transpose(1, -1).contiguous().view(*shape)
    size_factor = float(np.prod([x for i, x in enumerate(logits.size()) if i not in {0, dim}]).astype(np.float32))
    nll = torch.nn.functional.nll_loss(
            flatten(log_probs, (-1, args.levels)),
            flatten(x_quant, (-1,))) * size_factor
    return nll

# -------------------------------------------------------------------------------------------------------------
# From https://colab.research.google.com/github/tensorchiefs/dl_book/blob/master/chapter_04/nb_ch04_02.ipynb#scrollTo=n-nSWXYadTft