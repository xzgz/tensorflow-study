#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 afternoon 1:39
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : data_provider.py
# @IDE: PyCharm Community Edition
"""
Provide the training and testing data for shadow net
"""
import os.path as ops
import numpy as np
import copy, cv2, os
try:
    from cv2 import cv2
except ImportError:
    pass

from data_provider import base_data_provider


class TextDataset(base_data_provider.Dataset):
    """
        Implement a dataset class providing the image and it's corresponding text
    """
    def __init__(self, images, labels, imagenames, image_widths, shuffle=None, normalization=None):
        """

        :param images: image datasets [nums, H, W, C] 4D ndarray
        :param labels: label dataset [nums, :] 2D ndarray
        :param shuffle: if need shuffle the dataset, 'once_prior_train' represent shuffle only once before training
                        'every_epoch' represent shuffle the data every epoch
        :param imagenames:
        :param normalization: if need do normalization to the dataset,
                              'None': no any normalization
                              'divide_255': divide all pixels by 255
                              'divide_256': divide all pixels by 256
        """
        super(TextDataset, self).__init__()

        self.__normalization = normalization
        if self.__normalization not in [None, 'divide_255', 'divide_256']:
            raise ValueError('normalization parameter wrong')
        self.__images = self.normalize_images(images, self.__normalization)

        self.__labels = labels
        self.__imagenames = imagenames
        self.__image_widths = image_widths
        self._epoch_images = copy.deepcopy(self.__images)
        self._epoch_labels = copy.deepcopy(self.__labels)
        self._epoch_imagenames = copy.deepcopy(self.__imagenames)
        self._epoch_image_widths = copy.deepcopy(self.__image_widths)

        self.__shuffle = shuffle
        if self.__shuffle not in [None, 'once_prior_train', 'every_epoch']:
            raise ValueError('shuffle parameter wrong')
        if self.__shuffle == 'every_epoch' or 'once_prior_train':
            self._epoch_images, self._epoch_labels, self._epoch_imagenames, self._epoch_image_widths = \
                self.shuffle_images_labels(self._epoch_images, self._epoch_labels, self._epoch_imagenames, self._epoch_image_widths)

        self.__batch_counter = 0
        return

    @property
    def num_examples(self):
        """

        :return:
        """
        assert self.__images.shape[0] == self.__labels.shape[0]
        return self.__labels.shape[0]

    @property
    def images(self):
        """

        :return:
        """
        return self._epoch_images

    @property
    def labels(self):
        """

        :return:
        """
        return self._epoch_labels

    @property
    def imagenames(self):
        """

        :return:
        """
        return self._epoch_imagenames

    @property
    def image_widths(self):
        """

        :return:
        """
        return self._epoch_image_widths

    def next_batch(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        start = self.__batch_counter * batch_size
        end = (self.__batch_counter + 1) * batch_size
        self.__batch_counter += 1
        images_slice = self._epoch_images[start:end]
        labels_slice = self._epoch_labels[start:end]
        imagenames_slice = self._epoch_imagenames[start:end]
        # if overflow restart from the begining
        if images_slice.shape[0] != batch_size:
            self.__start_new_epoch()
            return self.next_batch(batch_size)
        else:
            return images_slice, labels_slice, imagenames_slice

    def __start_new_epoch(self):
        """

        :return:
        """
        self.__batch_counter = 0

        if self.__shuffle == 'every_epoch':
            self._epoch_images, self._epoch_labels, self._epoch_imagenames = self.shuffle_images_labels(
                self._epoch_images, self._epoch_labels, self._epoch_imagenames)
        else:
            pass
        return


class TextDataProvider(object):
    """
        Implement the text data provider for training and testing the shadow net
    """
    def __init__(self, dataset_dir, annotation_name, validation_set=None, validation_split=None, shuffle=None,
                 normalization=None):
        """

        :param dataset_dir: str, where you save the dataset one class on folder
        :param annotation_name: annotation name
        :param validation_set:
        :param validation_split: `float` or None float: chunk of `train set` will be marked as `validation set`.
                                 None: if 'validation set' == True, `validation set` will be
                                 copy of `test set`
        :param shuffle: if need shuffle the dataset, 'once_prior_train' represent shuffle only once before training
                        'every_epoch' represent shuffle the data every epoch
        :param normalization: if need do normalization to the dataset,
                              'None': no any normalization
                              'divide_255': divide all pixels by 255
                              'divide_256': divide all pixels by 256
                              'by_chanels': substract mean of every chanel and divide each
                                            chanel data by it's standart deviation
        """
        self.__dataset_dir = dataset_dir
        self.__validation_split = validation_split
        self.__shuffle = shuffle
        self.__normalization = normalization
        self.train = None
        assert ops.exists(dataset_dir)

        # add train dataset
        train_anno_path = ops.join(dataset_dir, annotation_name)
        assert ops.exists(train_anno_path)

        with open(train_anno_path, 'r') as anno_file:
            infos = np.array([tmp.strip() for tmp in anno_file.readlines()])
            imagenames = []
            for i in infos:
                imna = i
                if cv2.imread(imna, cv2.IMREAD_COLOR) is None:
                    print(imna + ' cannot be read!')
                    continue
                imagenames.append(imna)
            train_images = []
            train_image_widths = []
            train_labels = []
            for i in imagenames:
                image = cv2.imread(i, cv2.IMREAD_COLOR)
                h = image.shape[0]
                w = image.shape[1]
                new_w = int(w*32/h)
                if new_w <= 800:
                    pad_w = int(800 - new_w)
                    image = cv2.resize(image, (int(w*32/h), 32))
                    image = np.pad(
                        image, np.array([[0, 0], [0, pad_w], [0, 0]]), 'constant',
                        constant_values=np.array([[0, 0], [0, 0], [0, 0]]))
                    train_images.append(image)
                    train_image_widths.append(int(new_w/4))
                else:
                    image = cv2.resize(image, (800, 32))
                    train_images.append(image)
                    train_image_widths.append(int(800/4))
                train_labels.append('1')
                assert image.shape[1] == 800
            print('Read complete.')
            train_imagenames = []
            for n in imagenames:
                base_name = ops.basename(n)
                if n[-5] == 'T':
                    base_name = base_name[:-6]+'.jpg'
                train_imagenames.append(base_name)
            print(train_images[0].shape, train_images[1].shape)
            print('train image shape:', len(train_images))
            self.train = TextDataset(
                train_images, train_labels, imagenames=train_imagenames, image_widths=train_image_widths,
                shuffle=shuffle, normalization=normalization)
        anno_file.close()
        return

    def __str__(self):
        provider_info = 'Dataset_dir: {:s} contain training images: {:d} validation images: {:d} testing images: {:d}'.\
            format(self.__dataset_dir, self.train.num_examples, self.validation.num_examples, self.test.num_examples)
        return provider_info

    @property
    def dataset_dir(self):
        """

        :return:
        """
        return self.__dataset_dir

    @property
    def train_dataset_dir(self):
        """

        :return:
        """
        return self.__train_dataset_dir

    @property
    def test_dataset_dir(self):
        """

        :return:
        """
        return self.__test_dataset_dir
