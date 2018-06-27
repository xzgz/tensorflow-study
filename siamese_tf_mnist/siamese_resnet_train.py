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
# root = '/home/weiying1/hyg/tensorflow-study'
sys.path.insert(0, root)
os.chdir(root)

from siamese_tf_mnist import siamese_resnet_model


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_save_dir = 'model/mnist'
# model_save_dir = 'model/20180601_resnet_v2_imagenet_savedmodel/1527887769/variables'

# model_name = 'model.ckpt-resnet-ce'
# model_name = 'model.ckpt-resnet'
model_name = 'model.ckpt-resnet32-siamese-ce'
# model_name = 'model.ckpt-cnn-siamese-ce'
model_save_path = os.path.join(model_save_dir, model_name)

# snapshot = 'model.ckpt-resnet-92000'
snapshot = 'variables'

# model_snapshot_path = os.path.join(model_save_dir, snapshot)
model_snapshot_path = None

learning_rates = [0.01, 0.01, 0.001]
# learning_rates = [10.0, 1.0, 0.1]
# start_iterations = 92000
start_iterations = 0
max_iterations = 100000
boundaries = [20000, 80000]


def train_siamese_resnet():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.piecewise_constant(global_step, boundaries, learning_rates)
    siamese = siamese_resnet_model.Siamese(is_training=True)
    train_step = tf.train.GradientDescentOptimizer(lr).minimize(siamese.loss, global_step=global_step)
    saver = tf.train.Saver()

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

    sess = tf.InteractiveSession()
    if model_snapshot_path is None:
        tf.global_variables_initializer().run()
    else:
        print('*******************************************')
        print('Restore parameters from model {}'.format(model_snapshot_path))
        print('*******************************************')
        saver.restore(sess, save_path=model_snapshot_path)

    batch_size = 128
    tid = 360
    # batch_x1, batch_y1 = mnist.train.next_batch(128)
    # batch_x2, batch_y2 = mnist.train.next_batch(128)
    # batch_y = (batch_y1 == batch_y2).astype('float')
    batch_x1, batch_x2, batch_y = siamese_resnet_model.generate_train_samples(mnist, batch_size, positive_rate=0.5)
    distance, initial_loss = sess.run([siamese.distance, siamese.loss], feed_dict={
        siamese.x1: batch_x1,
        siamese.x2: batch_x2,
        siamese.y_: batch_y})

    correct_count = 0
    for i in range(100, test_images_num):  # len(test_images)
        tm = test_images[i]
        idn = siamese.single_sample_identity.eval({siamese.x1: siamese_resnet_model.format_single_sample(tm),
                                                   siamese.x2: gallery_image})
        if gallery_label[idn] == test_labels[i]:
            correct_count += 1
    accuracy = correct_count / (test_images_num-100)

    print('The initial loss:', initial_loss)
    print('Global step:', sess.run(global_step))
    print('Initial learning rate:', sess.run(lr))
    print('Initial accuracy: {:.4f}'.format(accuracy))
    distance, distance1, pre_id = sess.run(
        [siamese.distance, siamese.distance, siamese.single_sample_identity],
        feed_dict={siamese.x1: siamese_resnet_model.format_single_sample(test_images[tid]),
                   siamese.x2: gallery_image})
    print('distance:', distance, distance.shape, distance.dtype)
    print('True label: {}, predicted label: {}'.format(test_labels[tid], pre_id))
    print('distance1:\n', distance1)


    print('Start train...')
    for step in range(start_iterations, max_iterations):
        iterations = step+1
        batch_x1, batch_x2, batch_y = siamese_resnet_model.generate_train_samples(mnist, batch_size, positive_rate=0.5)

        _, loss_v, gs_v, lr_v, distance = sess.run(
            [train_step, siamese.loss, global_step, lr, siamese.distance],
            feed_dict={siamese.x1: batch_x1,
                       siamese.x2: batch_x2,
                       siamese.y_: batch_y})

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()
        if iterations % 500 == 0:
            # print('batch_y:\n', batch_y)
            # print('step %d: loss %.3f' % (iterations, loss_v))
            print('Global step: {:d}, iterations: {:d}, learning rate: {:.5f}, loss: {:.4f}'.format(
                gs_v, iterations, lr_v, loss_v))
            # print('distance:', distance)

        if iterations % 2000 == 0:
            saver.save(sess=sess, save_path=model_save_path, global_step=iterations)

            print('Start test...')
            correct_count = 0
            for i in range(100, 2100):
                tm = test_images[i]
                idn = siamese.single_sample_identity.eval({siamese.x1: siamese_resnet_model.format_single_sample(tm),
                                                           siamese.x2: gallery_image})
                if gallery_label[idn] == test_labels[i]:
                    correct_count += 1
            accuracy = correct_count / (2100-100)
            print('Test accuracy: {:.4f}'.format(accuracy))
        # if iterations % 10000 == 0:
            output1, output2, distance, distance1, pre_id = sess.run(
                [siamese.o1, siamese.o2, siamese.distance, siamese.distance, siamese.single_sample_identity],
                feed_dict={siamese.x1: siamese_resnet_model.format_single_sample(test_images[tid]),
                           siamese.x2: gallery_image})
            print('distance:', distance, distance.shape, distance.dtype)
            print('True label: {}, predicted label: {}'.format(test_labels[tid], pre_id))
            # for i, v in enumerate(output1):
                # print('output1_'+str(i)+':', output1[i])
                # print('output2_'+str(i)+':', output2[i])
            # print('distance1:\n', distance1)
        if iterations == max_iterations:
            print('Start test all...')
            correct_count = 0
            for i in range(100, test_images_num):
                tm = test_images[i]
                idn = siamese.single_sample_identity.eval({siamese.x1: siamese_resnet_model.format_single_sample(tm),
                                                           siamese.x2: gallery_image})
                if gallery_label[idn] == test_labels[i]:
                    correct_count += 1
            accuracy = correct_count / (test_images_num-100)
            print('Test accuracy: {:.4f}'.format(accuracy))
train_siamese_resnet()





