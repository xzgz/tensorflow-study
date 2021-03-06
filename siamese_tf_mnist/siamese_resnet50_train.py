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

from siamese_tf_mnist import siamese_resnet_model_50


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_save_dir = 'model/mnist'
# model_save_dir = 'model/20180601_resnet_v2_imagenet_savedmodel/1527887769/variables'

# model_name = 'model.ckpt-resnet-ce'
# model_name = 'model.ckpt-resnet'
model_name = 'model.ckpt-resnet50-siamese'
model_save_path = os.path.join(model_save_dir, model_name)

# snapshot = 'model.ckpt-resnet-92000'
snapshot = 'model.ckpt-resnet50'

model_snapshot_path = os.path.join(model_save_dir, snapshot)
# model_snapshot_path = None

# learning_rates = [0.01, 0.001, 0.0001]
learning_rates = [0.1, 0.01]
start_iterations = 20000
max_iterations = 30000
boundaries = [20000]


def train_siamese_resnet50():
    siamese = siamese_resnet_model_50.Siamese(is_training=True)
    sess = tf.InteractiveSession()
    # tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    if model_snapshot_path is None:
        tf.global_variables_initializer().run()
    else:
        print('*******************************************')
        print('Restore parameters from model {}'.format(model_snapshot_path))
        print('*******************************************')
        saver.restore(sess, save_path=model_snapshot_path)
    global_step = tf.Variable(start_iterations, name='global_step', trainable=False)
    # tf.global_variables_initializer().run()
    init_global_step = tf.variables_initializer([global_step])
    init_global_step.run()
    # global_step = tf.get_variable(name='global_step', shape=None, trainable=False, validate_shape=False)
    lr = tf.train.piecewise_constant(global_step, boundaries, learning_rates)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(siamese.loss, global_step=global_step)
    # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(siamese.loss)


    mnist = input_data.read_data_sets('data/mnist-data', one_hot=False)
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    test_images_num = len(test_images)
    print('There are {} test images.'.format(test_images_num))
    gallery_image = []
    gallery_label = []
    for i in range(100):
        if len(gallery_image) != 10:
            if test_labels[i] not in gallery_label:
                gallery_label.append(test_labels[i])
                gallery_image.append(test_images[i])
        else:
            break
    gallery_image = siamese_resnet_model_50.format_batch_resnet50(gallery_image)


    batch_size = 32
    print('Start initial test...')
    batch_x1, batch_y1 = mnist.train.next_batch(batch_size)
    batch_x2, batch_y2 = mnist.train.next_batch(batch_size)
    batch_y = (batch_y1 == batch_y2).astype('float')
    batch_x1, batch_x2 = siamese_resnet_model_50.format_batch_resnet50(batch_x1, batch_x2)
    initial_loss = sess.run(siamese.loss, feed_dict={
        siamese.x1: batch_x1,
        siamese.x2: batch_x2,
        siamese.y_: batch_y})
    correct_count = 0
    for i in range(100, 1100):  # len(test_images)
        tm = test_images[i]
        tm = siamese_resnet_model_50.format_batch_resnet50([tm])[0]
        tm = siamese_resnet_model_50.format_single_sample(tm)
        idn = siamese.single_sample_identity.eval({siamese.x1: tm,
                                                   siamese.x2: gallery_image})
        if gallery_label[idn] == test_labels[i]:
            correct_count += 1
    accuracy = correct_count / (1100-100)
    print('The initial loss:', initial_loss)
    print('Global step:', sess.run(global_step))
    print('Initial learning rate:', sess.run(lr))
    print('Initial accuracy: {:.4f}'.format(accuracy))


    print('Start train...')
    for step in range(start_iterations, max_iterations):
        iterations = step+1
        batch_x1, batch_y1 = mnist.train.next_batch(batch_size)
        batch_x2, batch_y2 = mnist.train.next_batch(batch_size)
        batch_y = (batch_y1 == batch_y2).astype('float')
        batch_x1, batch_x2 = siamese_resnet_model_50.format_batch_resnet50(batch_x1, batch_x2)

        _, loss_v, gs_v, lr_v = sess.run([train_step, siamese.loss, global_step, lr], feed_dict={
            siamese.x1: batch_x1,
            siamese.x2: batch_x2,
            siamese.y_: batch_y})

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()
        if iterations % 100 == 0:
            # print('batch_y:\n', batch_y)
            # print('step %d: loss %.3f' % (iterations, loss_v))
            print('Global step: {:d}, iterations: {:d}, learning rate: {:.5f}, loss: {:.4f}'.format(
                gs_v, iterations, lr_v, loss_v))

        if iterations % 500 == 0:
            saver.save(sess=sess, save_path=model_save_path, global_step=iterations)

            print('Start test...')
            correct_count = 0
            for i in range(100, 1100):
                tm = test_images[i]
                tm = siamese_resnet_model_50.format_batch_resnet50([tm])[0]
                tm = siamese_resnet_model_50.format_single_sample(tm)
                idn = siamese.single_sample_identity.eval({siamese.x1: tm,
                                                           siamese.x2: gallery_image})
                if gallery_label[idn] == test_labels[i]:
                    correct_count += 1
            accuracy = correct_count / (1100-100)
            print('Test accuracy: {:.4f}'.format(accuracy))
train_siamese_resnet50()




