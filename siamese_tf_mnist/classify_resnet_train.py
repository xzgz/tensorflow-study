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
import cv2
root = '/home/gysj/tensorflow-study'
sys.path.insert(0, root)
os.chdir(root)

from siamese_tf_mnist import siamese_resnet_model


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_save_dir = 'model/mnist'
# model_save_dir = 'model/20180601_resnet_v2_imagenet_savedmodel/1527887769/variables'

# model_name = 'model.ckpt-resnet-ce'
# model_name = 'model.ckpt-resnet'
model_name = 'model.ckpt-resnet32-classify'
model_save_path = os.path.join(model_save_dir, model_name)

# snapshot = 'model.ckpt-resnet-92000'
snapshot = 'variables'

# model_snapshot_path = os.path.join(model_save_dir, snapshot)
model_snapshot_path = None

# learning_rates = [0.01, 0.001, 0.0001]
learning_rates = [0.1, 0.01]
# start_iterations = 92000
start_iterations = 0
max_iterations = 40000
boundaries = [30000]


def train_classify_resnet():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.piecewise_constant(global_step, boundaries, learning_rates)
    siamese = siamese_resnet_model.Siamese(is_training=True)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(siamese.classify_loss, global_step=global_step)
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    if model_snapshot_path is None:
        tf.global_variables_initializer().run()
    else:
        print('Restore parameters from model {}'.format(model_snapshot_path))
        saver.restore(sess, save_path=model_snapshot_path)


    mnist = input_data.read_data_sets('data/mnist-data', one_hot=False)
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    test_images_num = len(test_images)
    print('There are {} test images.'.format(test_images_num))
    print('test_images:', test_images.shape, test_images.dtype)
    print('test_labels:', test_labels.shape, test_labels.dtype)
    test_labels = np.asarray(test_labels, np.int32)
    print('transformed test_labels:', test_labels.shape, test_labels.dtype)


    batch_size = 128
    batch_images, batch_labels = mnist.train.next_batch(batch_size)
    batch_labels = np.asarray(batch_labels, dtype=np.int32)
    initial_loss = sess.run(siamese.classify_loss, feed_dict={
        siamese.classify_images: batch_images,
        siamese.classify_labels: batch_labels})
    predict_labels = siamese.predicted_labels.eval({siamese.classify_images: test_images})
    correct_count = (predict_labels == test_labels).astype('int32')
    print('The initial loss:', initial_loss)
    print('Global step:', sess.run(global_step))
    print('Initial learning rate:', sess.run(lr))
    print('Initial accuracy: {:.4f}'.format(correct_count))


    print('Start train...')
    for step in range(start_iterations, max_iterations):
        iterations = step+1
        batch_images, batch_labels = mnist.train.next_batch(batch_size)
        batch_labels = np.asarray(batch_labels, dtype=np.int32)
        _, loss_v, gs_v, lr_v = sess.run([train_step, siamese.classify_loss, global_step, lr], feed_dict={
            siamese.classify_images: batch_images,
            siamese.classify_labels: batch_labels})

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()
        if iterations % 500 == 0:
            # print('batch_y:\n', batch_y)
            # print('step %d: loss %.3f' % (iterations, loss_v))
            print('Global step: {:d}, iterations: {:d}, learning rate: {:.5f}, loss: {:.4f}'.format(
                gs_v, iterations, lr_v, loss_v))

        if iterations % 2000 == 0:
            saver.save(sess=sess, save_path=model_save_path, global_step=iterations)

            print('Start test...')
            predict_labels = siamese.predicted_labels.eval({siamese.classify_images: test_images})
            correct_count = (predict_labels == test_labels).astype('int32')
            print('Test accuracy: {:.4f}'.format(correct_count))
train_classify_resnet()





