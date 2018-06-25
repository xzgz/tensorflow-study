
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from siamese_tf_mnist import resnet_model
import numpy as np


class Siamese:

    # Create model
    def __init__(self, is_training):
        self.x1 = tf.placeholder(tf.float32, [None, 784])
        self.x2 = tf.placeholder(tf.float32, [None, 784])
        self.y_ = tf.placeholder(tf.float32, [None])
        self.classify_images = tf.placeholder(tf.float32, [None, 784])
        self.classify_labels = tf.placeholder(tf.int32, [None])
        self.is_training = is_training

        self.o1 = self.cnn_model(self.x1, self.is_training, scope_reuse=False)
        self.o2 = self.cnn_model(self.x2, self.is_training, scope_reuse=True)
        # with self.model_variable_scope() as scope:
        #     self.o1 = self.network(self.x1)
        #     scope.reuse_variables()
        #     self.o2 = self.network(self.x2)
        print('self.o1 shape:', self.o1.shape)
        self.inner_product1 = tf.multiply(self.o1, self.o2)
        self.inner_product = tf.reduce_sum(self.inner_product1, axis=1)
        print('self.inner_product1 shape:', self.inner_product1.shape)
        self.loss = self.loss_cross_entropy(self.inner_product)
        self.single_sample_identity = tf.argmax(-self.inner_product, 0)
        # self.loss = self.loss_with_spring()
        # self.distance = self.pair_distance()
        # self.single_sample_identity = tf.argmax(-self.distance, 0)

        # self.classify_features = self.cnn_classify_model(self.classify_images, self.is_training, scope_reuse=False)
        # self.classify_loss = self.loss_classify(self.classify_features, self.classify_labels)
        # self.predicted_labels = self.classify_predict(self.classify_features)

    def model_variable_scope(self):
        return tf.variable_scope("siamese")

    def cnn_model(self, input_images, is_training, scope_reuse):
        inputs = tf.reshape(input_images, [-1, 1, 28, 28])
        # resnet50_mnist = resnet_model.Model(
        #     resnet_size=32,                         # resnet_size must be 6n+2, here n=5
        #     bottleneck=False,
        #     num_classes=64,
        #     num_filters=32,
        #     kernel_size=3,
        #     conv_stride=1,
        #     first_pool_size=2,
        #     first_pool_stride=2,
        #     block_sizes=[1],
        #     block_strides=[2],
        #     final_size=32,
        #     resnet_version=2,
        #     data_format='channels_first',
        #     dtype=tf.float32
        # )
        resnet50_mnist = resnet_model.Model(
            resnet_size=32,                         # resnet_size must be 6n+2, here n=5
            bottleneck=False,
            num_classes=32,
            num_filters=16,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=None,
            first_pool_stride=None,
            block_sizes=[5] * 3,
            block_strides=[1, 2, 2],
            final_size=64,
            resnet_version=2,
            data_format='channels_first',
            dtype=tf.float32
        )
        features = resnet50_mnist(inputs, is_training, scope_reuse)
        # params = tf.trainable_variables()
        # print(params)
        return features

    def cnn_classify_model(self, input_images, is_training, scope_reuse):
        inputs = tf.reshape(input_images, [-1, 1, 28, 28])
        resnet50_mnist = resnet_model.Model(
            resnet_size=32,                         # resnet_size must be 6n+2, here n=5
            bottleneck=False,
            num_classes=10,
            num_filters=16,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=None,
            first_pool_stride=None,
            block_sizes=[5] * 3,
            block_strides=[1, 2, 2],
            final_size=64,
            resnet_version=2,
            data_format='channels_first',
            dtype=tf.float32
        )
        features = resnet50_mnist(inputs, is_training, scope_reuse)
        return features

    def network(self, x):
        fc1 = self.fc_layer(x, 1024, "fc1")
        ac1 = tf.nn.relu(fc1)
        fc2 = self.fc_layer(ac1, 1024, "fc2")
        ac2 = tf.nn.relu(fc2)
        fc3 = self.fc_layer(ac2, 2, "fc3")
        return fc3

    def fc_layer(self, bottom, n_weight, name):
        assert len(bottom.get_shape()) == 2
        n_prev_weight = bottom.get_shape()[1]
        initer = tf.truncated_normal_initializer(stddev=0.01)
        W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=initer)
        b = tf.get_variable(name+'b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[n_weight], dtype=tf.float32))
        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
        return fc

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

    def loss_cross_entropy(self, inner_product):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_, logits=inner_product)
        loss = tf.reduce_mean(losses, name='siamese_loss')
        # tf.nn.softmax_cross_entropy_with_logits()
        return loss

    def loss_classify(self, features, labels):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=features)
        loss = tf.reduce_mean(losses, name='classify_loss')
        return loss

    def classify_predict(self, features):
        # It defaults to return a tensor of type tf.int64
        return tf.argmax(input=features, axis=1)

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


def generate_train_samples(mnist, batch_size, positive_rate):
    pos_cnt = int(batch_size*positive_rate)
    neg_cnt = batch_size - pos_cnt
    pos_num = 0
    neg_num = 0

    batch1 = []
    batch2 = []
    labels = []
    while pos_num != pos_cnt:
        batch_x1, batch_y1 = mnist.train.next_batch(100)
        batch_x2, batch_y2 = mnist.train.next_batch(100)
        batch_y = (batch_y1 == batch_y2)
        for i, v in enumerate(batch_y):
            if v:
                batch1.append(batch_x1[i])
                batch2.append(batch_x2[i])
                labels.append(batch_y[i])
            pos_num += 1
            if pos_num == pos_cnt:
                break
    while neg_num != neg_cnt:
        batch_x1, batch_y1 = mnist.train.next_batch(100)
        batch_x2, batch_y2 = mnist.train.next_batch(100)
        batch_y = (batch_y1 == batch_y2)
        for i, v in enumerate(batch_y):
            if not v:
                batch1.append(batch_x1[i])
                batch2.append(batch_x2[i])
                labels.append(batch_y[i])
            neg_num += 1
            if neg_num == neg_cnt:
                break
    batch_y = np.array(labels).astype('float32')

    return batch1, batch2, batch_y






