# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from libs.activations import lrelu

import argparse

# Import data
from tensorflow.examples.tutorials.mnist import input_data

# Import module
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# import scipy as sp
import math
import scipy.ndimage

FLAGS = None


# Layer functions

def weight_variable(shape):
    # initial = tf.truncated_normal(shape, stddev=0.1)
    initial = tf.random_uniform(shape, -1.0 / math.sqrt(shape[2]), 1.0 / math.sqrt(shape[2]))
    return tf.Variable(initial)


def bias_variable(shape):
    # initial = tf.constant(0.1, shape=shape)
    initial = tf.zeros(shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def deconv2d(x, W, shape):
    return tf.nn.conv2d_transpose(x, W, shape, strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# --------------------------------------------------------------------------------------------------------- #


def main(_):
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.float32, shape=[None, 28, 28])
    sess = tf.InteractiveSession()

    shape_layer = []

    # First convolution layer, the resulting image size would be 24 x 24
    # shape_layer.append(x.get_shape().as_list())

    im_in = tf.reshape(x, [-1, 28, 28, 1])
    im_in_blur = tf.reshape(y, [-1, 28, 28, 1])

    # import pdb; pdb.set_trace()
    # plt.figure()
    # plt.imshow(im_in_blur, cmap="gray")
    # plt.show(block=False)


    shape_layer.append(im_in.get_shape().as_list())

    w_conv1 = weight_variable([3, 3, 1, 10])
    b_conv1 = bias_variable([10])

    y_conv1 = lrelu(conv2d(im_in, w_conv1) + b_conv1)

    # Second convolution layer, the resulting image size would be 20 x 20
    shape_layer.append(y_conv1.get_shape().as_list())

    w_conv2 = weight_variable([3, 3, 10, 10])
    b_conv2 = bias_variable([10])

    y_conv2 = lrelu(conv2d(y_conv1, w_conv2) + b_conv2)

    # Third convolution layer, the resulting image size would be 16 x 16
    shape_layer.append(y_conv2.get_shape().as_list())

    w_conv3 = weight_variable([3, 3, 10, 10])
    b_conv3 = bias_variable([10])

    y_conv3 = lrelu(conv2d(y_conv2, w_conv3) + b_conv3)

    # Fourth convolution layer, the resulting image size would be 10 x 10
    shape_layer.append(y_conv3.get_shape().as_list())

    w_conv4 = weight_variable([3, 3, 10, 10])
    b_conv4 = bias_variable([10])

    y_conv4 = lrelu(conv2d(y_conv3, w_conv4) + b_conv4)

    # Fifth convolution layer, the resulting image size would be 6 x 6
    # shape_layer.append(y_conv4.get_shape().as_list())

    # w_conv5 = weight_variable([3, 3, 10, 10])
    # b_conv5 = bias_variable([10])

    # y_conv5 = lrelu(conv2d(y_conv4, w_conv5) + b_conv5)


    # Latent representation
    # z = y_conv5
    z = y_conv4

    # import pdb; pdb.set_trace()
    shape_layer.reverse()

    # Deconvolution layer, use the same weights as corresponding convolution layers

    b_deconv1 = bias_variable([10])
    b_deconv2 = bias_variable([10])
    b_deconv3 = bias_variable([10])
    # b_deconv4 = bias_variable([10])
    b_deconv4 = bias_variable([1])
    # b_deconv5 = bias_variable([1])

    y_deconv1 = lrelu(deconv2d(y_conv4, w_conv4, tf.pack(
        [tf.shape(x)[0], shape_layer[0][1], shape_layer[0][2], shape_layer[0][3]])) + b_deconv1)

    y_deconv2 = lrelu(deconv2d(y_deconv1, w_conv3, tf.pack(
        [tf.shape(x)[0], shape_layer[1][1], shape_layer[1][2], shape_layer[1][3]])) + b_deconv2)

    y_deconv3 = lrelu(deconv2d(y_deconv2, w_conv2, tf.pack(
        [tf.shape(x)[0], shape_layer[2][1], shape_layer[2][2], shape_layer[2][3]])) + b_deconv3)

    y_deconv4 = lrelu(deconv2d(y_deconv3, w_conv1, tf.pack(
        [tf.shape(x)[0], shape_layer[3][1], shape_layer[3][2], shape_layer[3][3]])) + b_deconv4)

    # y_deconv5 = lrelu(deconv2d(y_deconv4, w_conv1, tf.pack([tf.shape(x)[0], shape_layer[4][1], shape_layer[4][2], shape_layer[4][3]]) ) + b_deconv5)

    # Now the output has been reconstructed through the network
    # y_out = y_deconv5
    y_out = y_deconv4

    # Define loss and optimizer
    loss = tf.reduce_sum(tf.square(y_out - im_in_blur))
    train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

    sess.run(tf.initialize_all_variables())

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    print(mnist.train.images.shape)
    mean_img = np.mean(mnist.train.images, axis=0)

    batch_size = 100
    n_epoch = 500
    for epoch_i in range(n_epoch):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            batch_xs_train = np.array([img - mean_img for img in batch_xs])
            # import pdb; pdb.set_trace()
            batch_xs_train_blur = scipy.ndimage.gaussian_filter(np.reshape(batch_xs_train, (-1, 28, 28)), sigma=1)
            sess.run(train_step, feed_dict={x: batch_xs_train, y: batch_xs_train_blur})
        print("epoch: " + str(epoch_i), sess.run(loss, feed_dict={x: batch_xs_train, y: batch_xs_train_blur}))

    # Plot example reconstructions
    n_examples = 10
    test_xs, _ = mnist.test.next_batch(n_examples)
    test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(y_out, feed_dict={x: test_xs_norm})
    print(recon.shape)
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(test_xs[example_i, :], (28, 28)))
        axs[1][example_i].imshow(
            np.reshape(
                np.reshape(recon[example_i, ...], (784,)) + mean_img,
                (28, 28)))
    fig.show()
    plt.draw()
    plt.waitforbuttonpress()

    # Functions for visualizing the response of activation layrers

    def getActivations(layers, stimuli, fig_num=1):
        units_layers = []
        for layer in layers:
            units = layer.eval(session=sess, feed_dict={x: np.reshape(stimuli, [1, 784], order='F')})
            print(units.shape)
            units_layers.append(units)
        plotNNFilter(units_layers, fig_num)

    def plotNNFilter(units_layers, fig_num=1):
        # import pdb; pdb.set_trace()
        filters_maxNum = np.max([units.shape[3] for units in units_layers])
        # fig_act = plt.figure(fig_num, figsize=(20,20))
        fig_act, axs = plt.subplots(len(units_layers), filters_maxNum, figsize=(25, 15))
        for layer_idx, units in enumerate(units_layers):
            for filter_i in range(0, units.shape[3]):
                plt.title('Filter ' + str(filter_i))
                axs[layer_idx][filter_i].imshow(units[0, :, :, filter_i], interpolation="nearest", cmap="gray")

        # for layer_idx ,units in enumerate(units_layers):
        #    filters = units.shape[3]
        #    plot_num = 1
        #    for i in range(0,filters):
        #        #import pdb; pdb.set_trace()
        #        plt.subplot( len(units_layers), filters_maxNum, plot_num + layer_idx*filters_maxNum)
        #        plt.title('Filter ' + str(i))
        #        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
        #        plot_num += 1

        fig_act.tight_layout()
        fig_act.show()
        plt.waitforbuttonpress()


        # Show activation functions

        # imageToUse = mnist.test.images[0]
        # plt.imshow(np.reshape(imageToUse,[28,28]), interpolation="nearest", cmap="gray")

        # getActivations([y_conv1, y_conv2, y_conv3, y_conv4], imageToUse, 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/data',
                        help='Directory for storing data')
    FLAGS = parser.parse_args()
    tf.app.run()
