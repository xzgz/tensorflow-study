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
sys.path.insert(0, '/home/gysj/tensorflow-study')
os.chdir('/home/gysj/tensorflow-study/siamese_tf_mnist')

# import helpers
from siamese_tf_mnist import inference
from siamese_tf_mnist import visualize

model_save_dir = '../model/mnist'
model_name = 'model.ckpt'
snapshot = 'model.ckpt-10000'
model_save_path = os.path.join(model_save_dir, model_name)
model_snapshot_path = os.path.join(model_save_dir, snapshot)


def train_siamese():
    # prepare data and tf.session
    mnist = input_data.read_data_sets('../data/mnist-data', one_hot=False)
    sess = tf.InteractiveSession()

    siamese = inference.siamese(tf.estimator.ModeKeys.TRAIN)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()

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

    # start training
    for step in range(10000):
        iterations = step+1
        batch_x1, batch_y1 = mnist.train.next_batch(128)
        batch_x2, batch_y2 = mnist.train.next_batch(128)
        batch_y = (batch_y1 == batch_y2).astype('float')

        _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
            siamese.x1: batch_x1,
            siamese.x2: batch_x2,
            siamese.y_: batch_y})

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()
        if iterations % 100 == 0:
            print('step %d: loss %.3f' % (iterations, loss_v))

        if iterations % 1000 == 0:
            saver.save(sess=sess, save_path=model_save_path, global_step=iterations)
            #
            # print('Start test...')
            # correct_count = 0
            # for i in range(100, len(test_images)):
            #     tm = test_images[i]
            #     idn = siamese.single_sample_identity.eval({siamese.x1: inference.format_single_sample(tm),
            #                                                siamese.x2: gallery_image})
            #     if gallery_label[idn] == test_labels[i]:
            #         correct_count += 1
            # accuracy = correct_count / (len(test_images)-100)
            # print('Test accuracy:', accuracy)
# train_siamese()


def predict_single_sample():
    mnist = input_data.read_data_sets('../data/mnist-data', one_hot=False)
    sess = tf.InteractiveSession()

    # setup siamese network
    siamese = inference.siamese(tf.estimator.ModeKeys.PREDICT)
    saver = tf.train.Saver()
    saver.restore(sess, save_path=model_save_path)
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    print('test_images:', test_images.shape, test_images.dtype)
    print('test_labels:', test_labels.shape, test_labels.dtype)
    gallery_image = []
    gallery_label = []
    for i in range(100):
        if len(gallery_image) != 10:
            if test_labels[i] not in gallery_label:
                gallery_label.append(test_labels[i])
                gallery_image.append(test_images[i])
        else:
            break
    # print(type(gallery_image[1]))
    # gallery_image = np.array(gallery_image)
    # print(i, len(gallery_label), type(gallery_label[0]))
    # print(gallery_label)
    # print(type(gallery_image[1]))
    # print(gallery_image.shape)
    # visualize.show_data(gallery_image.reshape([-1, 28, 28]), 'gallery_image')
    # plt.show()

    tm = test_images[121]
    distance = siamese.distance.eval({siamese.x1: inference.format_single_sample(tm),
                                      siamese.x2: gallery_image})
    idn = siamese.single_sample_identity.eval({siamese.x1: inference.format_single_sample(tm),
                                               siamese.x2: gallery_image})
    print(idn, type(idn))
    print('predict label:', gallery_label[idn])
    print('true label:', test_labels[121])
    print('distance:', distance)
    print(tm.reshape([28, 28])[14, :])
    plt.imshow((tm.reshape([28, 28])*255).astype('uint8'), cmap='gray')
    plt.show()
# predict_single_sample()


def test_accuracy():
    mnist = input_data.read_data_sets('../data/mnist-data', one_hot=False)
    sess = tf.InteractiveSession()

    # setup siamese network
    siamese = inference.siamese(tf.estimator.ModeKeys.PREDICT)
    saver = tf.train.Saver()
    saver.restore(sess, save_path=model_snapshot_path)
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    print('test_images:', test_images.shape, test_images.dtype)
    print('test_labels:', test_labels.shape, test_labels.dtype)
    gallery_image = []
    gallery_label = []
    for i in range(100):
        if len(gallery_image) != 10:
            if test_labels[i] not in gallery_label:
                gallery_label.append(test_labels[i])
                gallery_image.append(test_images[i])
        else:
            break

    correct_count = 0
    for i in range(100, len(test_images)):
        tm = test_images[i]
        idn = siamese.single_sample_identity.eval({siamese.x1: inference.format_single_sample(tm),
                                                   siamese.x2: gallery_image})
        if gallery_label[idn] == test_labels[i]:
            correct_count += 1
        if (i+1) % 1000 == 0:
            print('Test {:d} images'.format(i+1))
    accuracy = correct_count / (len(test_images)-100)
    print('accuracy:', accuracy)
test_accuracy()




