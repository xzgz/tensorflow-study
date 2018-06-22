#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 afternoon 7:47
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : write_text_features.py
# @IDE: PyCharm Community Edition
"""
Write text features into tensorflow records
"""
import os
import os.path as ops
import argparse
import numpy as np
import cv2, sys
try:
    from cv2 import cv2
except ImportError:
    pass
# root = '/media/xzgz/Ubuntu/Ubuntu/Caffe/eclipse-caffe/crnn/crnn-tensorflow'
root = '/home/weiying1/hyg/icpr/CRNN_Tensorflow'
os.chdir(root)
sys.path.insert(0, root)
from data_provider import data_provider_icprtest
from local_utils import data_utils

train_image_width = 800
# train_image_width = 200
# train_image_width = 48
data_provider = data_provider_icprtest


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Where you store the dataset',
                        default='/home/weiying1/hyg/data/icpr/icpr_mtwi_test_pic_txt2')
#     parser.add_argument('--save_dir', type=str, help='Where you store tfrecords',
#                         default='/media/xzgz/Ubuntu/Ubuntu/Caffe/eclipse-caffe/crnn/tfrecords2')
#     parser.add_argument('--dataset_dir', type=str, help='Where you store the dataset',
#                         default='/home/weiying1/hyg/data/icpr/mnt/ramdisk/max')
#     parser.add_argument('--save_dir', type=str, help='Where you store tfrecords',
#                         default='/home/weiying1/hyg/data/icpr/tfrecords-gech2')
#     parser.add_argument('--anno_name', type=str, help='File listing test pictures name',
#                         default='label.txt')

    parser.add_argument('--save_dir', type=str, help='Where you store tfrecords',
                        default='/home/weiying1/hyg/data/icpr/icpr_test2')
    parser.add_argument('--anno_name', type=str, help='File listing test pictures name',
                        default='label.txt')

    return parser.parse_args()


def write_features(dataset_dir, save_dir, anno_name):
    """

    :param dataset_dir:
    :param save_dir:
    :return:
    """
    if not ops.exists(save_dir):
        os.makedirs(save_dir)

    print('Initialize the dataset provider ......')
    provider = data_provider.TextDataProvider(dataset_dir=dataset_dir, annotation_name=anno_name,
                                              validation_set=True, validation_split=0.001, shuffle=None,
                                              normalization=None)
    print('Dataset provider intialize complete')

    feature_io = data_utils.TextFeatureIO()

    # write train tfrecords
    print('Start writing training tf records')
    train_images_temp = provider.train.images
    train_image_widths = provider.train.image_widths
    train_images = []
    for index, image in enumerate(train_images_temp):
        train_images.append(bytes(list(np.reshape(image, [train_image_width*32*3]))))
    print(len(train_images))
    train_labels = provider.train.labels
    train_imagenames = provider.train.imagenames
    train_tfrecord_path = ops.join(save_dir, anno_name[:-4]+'.tfrecords')  # 'train_feature.tfrecords'
    train_class_num = feature_io.writer.write_features(
        tfrecords_path=train_tfrecord_path, labels=train_labels, images=train_images, imagenames=train_imagenames,
        image_widths=train_image_widths)
    print('training class_num: ', train_class_num)

    # # write test tfrecords
    # print('Start writing testing tf records')
    # test_images_temp = provider.test.images
    # test_image_widths = provider.test.image_widths
    # test_images = []
    # for index, image in enumerate(test_images_temp):
    #     test_images.append(bytes(list(np.reshape(image, [train_image_width*32*3]))))
    # print(len(test_images))
    # test_labels = provider.test.labels
    # test_imagenames = provider.test.imagenames
    # test_tfrecord_path = ops.join(save_dir, 'test_feature.tfrecords')
    # test_class_num = feature_io.writer.write_features(
    #     tfrecords_path=test_tfrecord_path, labels=test_labels, images=test_images, imagenames=test_imagenames,
    #     image_widths=test_image_widths)
    # print('test num_class: ', test_class_num)

    # write val tfrecords
    # val_images_temp = provider.validation.images
    # val_image_widths = provider.validation.image_widths
    # val_images = []
    # for index, image in enumerate(val_images_temp):
    #     val_images.append(bytes(list(np.reshape(image, [train_image_width*32*3]))))
    # print(len(val_images))
    # val_labels = provider.validation.labels
    # val_imagenames = provider.validation.imagenames
    # val_tfrecord_path = ops.join(save_dir, 'validation_feature.tfrecords')
    # val_class_num = feature_io.writer.write_features(
    #     tfrecords_path=val_tfrecord_path, labels=val_labels, images=val_images, imagenames=val_imagenames,
    #     image_widths=val_image_widths)
    # print('val num_class: ', val_class_num)

    return


if __name__ == '__main__':
    # init args
    args = init_args()
    if not ops.exists(args.dataset_dir):
        raise ValueError('Dataset {:s} doesn\'t exist'.format(args.dataset_dir))

    # write tf records
    write_features(dataset_dir=args.dataset_dir, save_dir=args.save_dir, anno_name=args.anno_name)
