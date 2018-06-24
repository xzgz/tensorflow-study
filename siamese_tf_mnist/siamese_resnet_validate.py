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
sys.path.insert(0, '/home/gysj/tensorflow-study')
os.chdir('/home/gysj/tensorflow-study')

from siamese_tf_mnist import siamese_resnet_model_50

# model_save_dir = 'model/mnist'
model_save_dir = 'model/20180601_resnet_v2_imagenet_savedmodel/1527887769/variables'
# snapshot = 'model.ckpt-resnet-98000'
snapshot = 'variables'
model_snapshot_path = os.path.join(model_save_dir, snapshot)


def validate_accuracy():
    mnist = input_data.read_data_sets('data/mnist-data', one_hot=False)
    sess = tf.InteractiveSession()
    siamese = siamese_resnet_model_50.Siamese(is_training=False)
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()

    print('Restore parameters from model {}'.format(model_snapshot_path))
    saver.restore(sess, save_path=model_snapshot_path)
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    test_images_num = len(test_images)
    print('There are {} test images.'.format(test_images_num))
    print('test_images:', test_images.shape, test_images.dtype)
    print('test_labels:', test_labels.shape, test_labels.dtype)
    gallery_image = []
    gallery_label = []
    for i in range(100):
        if len(gallery_image) != 10:
            if test_labels[i] not in gallery_label:
                im = test_images[i].reshape([28, 28])
                im = cv2.resize(im, (224, 224))
                im = np.tile(im, (3, 1, 1))
                gallery_image.append(im)
                gallery_label.append(test_labels[i])
        else:
            break

    correct_count = 0
    for i in range(100, test_images_num):
        tm = test_images[i]
        tm = tm.reshape([28, 28])
        tm = cv2.resize(tm, (224, 224))
        tm = np.tile(tm, (3, 1, 1))
        idn = siamese.single_sample_identity.eval({siamese.x1: siamese_resnet_model_50.format_single_sample(tm),
                                                   siamese.x2: gallery_image})
        if gallery_label[idn] == test_labels[i]:
            correct_count += 1
        if (i+1) % 1000 == 0:
            print('Test {:d} images'.format(i+1))
    accuracy = correct_count / (test_images_num-100)
    print('accuracy:', accuracy)
# validate_accuracy()


def predict_single_sample():
    mnist = input_data.read_data_sets('data/mnist-data', one_hot=False)
    sess = tf.InteractiveSession()
    siamese = siamese_resnet_model_50.Siamese(is_training=False)
    saver = tf.train.Saver()
    # tf.global_variables_initializer().run()
    #
    print('Restore parameters from model {}'.format(model_snapshot_path))
    saver.restore(sess, save_path=model_snapshot_path)
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    test_images_num = len(test_images)
    print('There are {} test images.'.format(test_images_num))
    print('test_images:', test_images.shape, test_images.dtype)
    print('test_labels:', test_labels.shape, test_labels.dtype)
    gallery_image = []
    gallery_label = []
    for i in range(100):
        if len(gallery_image) != 10:
            if test_labels[i] not in gallery_label:
                im = test_images[i].reshape([28, 28])
                im = cv2.resize(im, (224, 224))
                im = np.tile(im, (3, 1, 1))
                gallery_image.append(im)
                gallery_label.append(test_labels[i])
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
    tm = tm.reshape([28, 28])
    tm = cv2.resize(tm, (224, 224))
    tm = np.tile(tm, (3, 1, 1))
    distance = siamese.distance.eval({siamese.x1: siamese_resnet_model_50.format_single_sample(tm),
                                      siamese.x2: gallery_image})
    idn = siamese.single_sample_identity.eval({siamese.x1: siamese_resnet_model_50.format_single_sample(tm),
                                               siamese.x2: gallery_image})
    print(idn, type(idn))
    print('predict label:', gallery_label[idn])
    print('true label:', test_labels[121])
    print('distance:', distance)
    # print(tm.reshape([28, 28])[14, :])
    # plt.imshow((tm.reshape([28, 28])*255).astype('uint8'), cmap='gray')
    # plt.show()
predict_single_sample()


def some_test():
    import cv2
    mnist = input_data.read_data_sets('data/mnist-data', one_hot=False)
    test_images = mnist.test.images
    test_labels = mnist.test.labels
    test_images_num = len(test_images)
    print('There are {} test images.'.format(test_images_num))
    print('test_images:', test_images.shape, test_images.dtype)
    print('test_labels:', test_labels.shape, test_labels.dtype)
    gallery_image = []
    gallery_label = []
    for i in range(100):
        if len(gallery_image) != 10:
            if test_labels[i] not in gallery_label:
                im = test_images[i].reshape([28, 28])
                im = cv2.resize(im, (224, 224))
                im = np.tile(im, (3, 1, 1))
                gallery_image.append(im)
                gallery_label.append(test_labels[i])
        else:
            break
    print('gallery_image shape:', np.array(gallery_image).shape)
    tm = test_images[121]
    tm = tm.reshape([28, 28])
    tm = cv2.resize(tm, (224, 224))
    tm = np.tile(tm, (3, 1, 1))
    tm = np.tile(tm, (10, 1, 1, 1))
    # tm = tm.transpose((1, 2, 0))
    print('tm:', tm.shape, tm.dtype)
    # print(tm[200, :, 1])
    plt.imshow(tm[3].transpose((1, 2, 0)))
    plt.show()
# some_test()



