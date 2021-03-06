#!/usr/bin/env python
""" Siamese implementation using Tensorflow with MNIST example.
This siamese network embeds a 28x28 image (a point in 784D)
into a point in 2D.

By Youngwook Paul Kwon (young at berkeley.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import sys
root = '/home/gysj/tensorflow-study'
sys.path.insert(0, root)
os.chdir(root)

# import helpers
from siamese_tf_mnist import inference

model_save_dir = 'model/mnist'
model_name = 'model.ckpt'
model_save_path = os.path.join(model_save_dir, model_name)
# learning_rates = [0.01, 0.001, 0.0001]
learning_rates = [0.01, 0.001]
boundaries = [40000]


def train_siamese():
    # prepare data and tf.session
    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.piecewise_constant(global_step, boundaries, learning_rates)
    siamese = inference.siamese(tf.estimator.ModeKeys.TRAIN)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(siamese.loss, global_step=global_step)
    saver = tf.train.Saver()

    mnist = input_data.read_data_sets('data/mnist-data', one_hot=False)
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    gallery_image = []
    gallery_label = []
    for i in range(100):
        if len(gallery_image) != 10:
            if test_labels[i] not in gallery_label:
                gallery_label.append(test_labels[i])
                gallery_image.append(test_images[i])
        else:
            break

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    batch_x1, batch_y1 = mnist.train.next_batch(128)
    batch_x2, batch_y2 = mnist.train.next_batch(128)
    batch_y = (batch_y1 == batch_y2).astype('float')
    initial_loss = sess.run(siamese.loss, feed_dict={
        siamese.x1: batch_x1,
        siamese.x2: batch_x2,
        siamese.y_: batch_y})
    print('The initial loss:', initial_loss)
    print('Global step:', sess.run(global_step))
    print('Initial learning rate:', sess.run(lr))

    print('Start train...')
    for step in range(80000):
        iterations = step+1
        batch_x1, batch_y1 = mnist.train.next_batch(128)
        batch_x2, batch_y2 = mnist.train.next_batch(128)
        batch_y = (batch_y1 == batch_y2).astype('float')

        _, loss_v, gs_v, lr_v = sess.run([train_step, siamese.loss, global_step, lr], feed_dict={
            siamese.x1: batch_x1,
            siamese.x2: batch_x2,
            siamese.y_: batch_y})

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()
        if iterations % 100 == 0:
            # print('step %d: loss %.3f' % (iterations, loss_v))
            print('Global step: {:d}, iterations: {:d}, learning rate: {:.5f}, loss: {:.4f}'.format(
                gs_v, iterations, lr_v, loss_v))

        if iterations % 1000 == 0:
            saver.save(sess=sess, save_path=model_save_path, global_step=iterations)

            print('Start test...')
            correct_count = 0
            for i in range(100, len(test_images)):
                tm = test_images[i]
                idn = siamese.single_sample_identity.eval({siamese.x1: inference.format_single_sample(tm),
                                                           siamese.x2: gallery_image})
                if gallery_label[idn] == test_labels[i]:
                    correct_count += 1
            accuracy = correct_count / (len(test_images)-100)
            print('Test accuracy: {:.4f}'.format(accuracy))
train_siamese()





