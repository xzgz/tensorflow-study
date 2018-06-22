
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from official.resnet import resnet_model
import numpy as np


class Siamese:

    # Create model
    def __init__(self, is_training):
        self.x1 = tf.placeholder(tf.float32, [None, 784])
        self.x2 = tf.placeholder(tf.float32, [None, 784])
        self.is_training = is_training

        with self.model_variable_scope() as scope:
            self.o1 = self.cnn_model(self.x1, self.is_training, self.model_variable_scope)
            scope.reuse_variables()
            self.o2 = self.cnn_model(self.x2, self.is_training, self.model_variable_scope)

        # Create loss
        self.y_ = tf.placeholder(tf.float32, [None])
        self.loss = self.loss_with_spring()
        self.distance = self.pair_distance()
        self.single_sample_identity = tf.argmax(-self.distance, 0)

    def model_variable_scope(self):
        return tf.variable_scope("siamese")

    def cnn_model(self, input_images, is_training, model_variable_scope):
        inputs = tf.reshape(input_images, [-1, 1, 28, 28])
        resnet50_mnist = resnet_model.Model(
            resnet_size=32,                         # resnet_size must be 6n+2, here n=5
            bottleneck=False,
            num_classes=64,
            num_filters=32,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=2,
            first_pool_stride=2,
            block_sizes=[1],
            block_strides=[2],
            final_size=32,
            resnet_version=2,
            data_format='channels_first',
            dtype=tf.float32
        )
        features = resnet50_mnist(inputs, is_training, model_variable_scope)
        params=tf.trainable_variables()
        print(params)

        return features

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def loss_with_step(self):
        margin = 5.0
        labels_t = self.y_
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")          # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2+1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
        neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C, eucd)), name="Ny_C-eucd")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss

    def pair_distance(self):
        # length = tf.pow(self.o1, 2)+tf.pow(self.o2, 2)
        # length = tf.reduce_sum(length, 1)
        # length = tf.sqrt(length+1e-6)
        eucd = tf.pow(tf.subtract(self.o1, self.o2), 2)
        eucd = tf.reduce_sum(eucd, 1)
        eucd = tf.sqrt(eucd+1e-6, name="eucd")
        distance = eucd
        # distance = tf.divide(eucd, length)
        return distance


def format_single_sample(one_data):
    return np.tile(one_data, (10, 1))



