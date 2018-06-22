#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-25 afternoon 3:56
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : test_shadownet.py
# @IDE: PyCharm Community Edition
"""
Test shadow net script
"""
import os.path as ops
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import numpy as np
import math, sys
import os
# root = '/media/xzgz/Ubuntu/Ubuntu/Caffe/eclipse-caffe/crnn/crnn-tensorflow'
root = '/home/weiying1/hyg/icpr/CRNN_Tensorflow'
os.chdir(root)
sys.path.insert(0, root)
from local_utils import data_utils_im
from crnn_model import crnn_model_im
from global_configuration import config


data_utils = data_utils_im
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# snapshot_name = '/shadownet_2018-05-10-10-49-36.ckpt-60000'
# snapshot_name = '/shadownet_2018-05-10-10-49-36.ckpt-100000'
# snapshot_name = '/shadownet_2018-05-10-10-49-36.ckpt-140000'
# snapshot_name = '/shadownet_2018-05-10-10-49-36.ckpt-160000'
# snapshot_name = '/shadownet_2018-05-10-10-49-36.ckpt-180000'
# snapshot_name = '/shadownet_2018-05-10-10-49-36.ckpt-200000'
# snapshot_name = '/shadownet_2018-05-10-10-49-36.ckpt-220000'
# snapshot_name = '/shadownet_2018-05-10-10-49-36.ckpt-240000'
# snapshot_name = '/shadownet_2018-05-10-10-49-36.ckpt-260000'
# snapshot_name = '/shadownet_2018-05-10-10-49-36.ckpt-280000'
# snapshot_name = '/shadownet_2018-05-10-10-49-36.ckpt-300000'
# snapshot_name = '/shadownet_2018-05-10-10-49-36.ckpt-320000'
# snapshot_name = '/shadownet_2018-05-10-10-49-36.ckpt-340000'
# snapshot_name = '/shadownet_2018-05-10-10-49-36.ckpt-360000'
# snapshot_name = '/shadownet_2018-05-10-10-49-36.ckpt-380000'
# snapshot_name = '/shadownet_2018-05-10-10-49-36.ckpt-400000'
# snapshot_name = '/shadownet_2018-05-17-16-32-28.ckpt-420000'
# snapshot_path = '/home/weiying1/hyg/icpr/CRNN_Tensorflow/model/shadownet-im'

# shadownet-im
# snapshot_name = '/shadownet_2018-05-17-14-01-49.ckpt-440000'
# snapshot_name = '/shadownet_2018-05-17-14-01-49.ckpt-460000'
# snapshot_name = '/shadownet_2018-05-17-14-01-49.ckpt-480000'
# snapshot_name = '/shadownet_2018-05-17-14-01-49.ckpt-500000'
# snapshot_name = '/shadownet_2018-05-17-14-01-49.ckpt-520000'
# snapshot_name = '/shadownet_2018-05-17-14-01-49.ckpt-540000'
# snapshot_path = '/home/weiying1/hyg/icpr/CRNN_Tensorflow/model/shadownet-im'

# shadownet-im-3
# snapshot_name = '/shadownet_2018-05-17-16-32-28.ckpt-440000'
# snapshot_name = '/shadownet_2018-05-17-16-32-28.ckpt-460000'
# snapshot_name = '/shadownet_2018-05-17-16-32-28.ckpt-480000'
# snapshot_name = '/shadownet_2018-05-17-16-32-28.ckpt-500000'
# snapshot_name = '/shadownet_2018-05-17-16-32-28.ckpt-520000'
# snapshot_name = '/shadownet_2018-05-17-16-32-28.ckpt-540000'
# snapshot_name = '/shadownet_2018-05-19-19-10-10.ckpt-560000'
# snapshot_name = '/shadownet_2018-05-19-19-10-10.ckpt-580000'
# snapshot_path = '/home/weiying1/hyg/icpr/CRNN_Tensorflow/model/shadownet-im-3'

# snapshot_name = '/shadownet_2018-05-19-19-45-19.ckpt-25000'
# snapshot_name = '/shadownet_2018-05-19-19-45-19.ckpt-15000'
# snapshot_name = '/shadownet_2018-05-19-19-45-19.ckpt-20000'
# snapshot_name = '/shadownet_2018-05-19-19-45-19.ckpt-30000'
# snapshot_name = '/shadownet_2018-05-19-19-45-19.ckpt-35000'
# snapshot_name = '/shadownet_2018-05-20-21-40-17.ckpt-40000'
# snapshot_name = '/shadownet_2018-05-20-21-40-17.ckpt-50000'
# snapshot_name = '/shadownet_2018-05-20-21-40-17.ckpt-54000'
# snapshot_path = '/home/weiying1/hyg/icpr/CRNN_Tensorflow/model/shadownet-im-wchn'

# snapshot_name = '/shadownet_2018-05-20-13-15-24.ckpt-1000'
# snapshot_name = '/shadownet_2018-05-20-13-15-24.ckpt-2000'
# snapshot_name = '/shadownet_2018-05-20-13-15-24.ckpt-3000'
# snapshot_name = '/shadownet_2018-05-20-13-15-24.ckpt-4000'
# snapshot_name = '/shadownet_2018-05-20-13-15-24.ckpt-8000'
# snapshot_name = '/shadownet_2018-05-20-13-15-24.ckpt-10000'
# snapshot_name = '/shadownet_2018-05-20-13-15-24.ckpt-12000'
# snapshot_name = '/shadownet_2018-05-20-13-15-24.ckpt-14000'
# snapshot_name = '/shadownet_2018-05-20-13-15-24.ckpt-16000'
# snapshot_name = '/shadownet_2018-05-20-13-15-24.ckpt-18000'
# snapshot_name = '/shadownet_2018-05-20-13-15-24.ckpt-20000'
# snapshot_name = '/shadownet_2018-05-20-13-15-24.ckpt-22000'
# snapshot_name = '/shadownet_2018-05-20-13-15-24.ckpt-24000'
# snapshot_name = '/shadownet_2018-05-20-13-15-24.ckpt-25000'
# snapshot_name = '/shadownet_2018-05-20-13-15-24.ckpt-30000'
# snapshot_name = '/shadownet_2018-05-21-17-58-29.ckpt-31000'
# snapshot_name = '/shadownet_2018-05-22-01-23-50.ckpt-30000'
# snapshot_name = '/shadownet_2018-05-22-01-23-50.ckpt-35000'
# snapshot_name = '/shadownet_2018-05-22-01-23-50.ckpt-37000'
# snapshot_name = '/shadownet_2018-05-22-01-23-50.ckpt-39000'
# snapshot_name = '/shadownet_2018-05-22-01-23-50.ckpt-40000'
# snapshot_name = '/shadownet_2018-05-22-01-23-50.ckpt-41000'
# snapshot_name = '/shadownet_2018-05-22-01-23-50.ckpt-42000'
# snapshot_name = '/shadownet_2018-05-22-01-23-50.ckpt-43000'
# snapshot_path = '/home/weiying1/hyg/icpr/CRNN_Tensorflow/model/shadownet-im-wchn-2'

# snapshot_name = '/shadownet_2018-05-21-18-30-49.ckpt-1000'
# snapshot_name = '/shadownet_2018-05-21-18-30-49.ckpt-2000'
# snapshot_name = '/shadownet_2018-05-21-18-30-49.ckpt-3000'
# snapshot_name = '/shadownet_2018-05-21-18-30-49.ckpt-4000'
# snapshot_name = '/shadownet_2018-05-21-18-30-49.ckpt-5000'
# snapshot_name = '/shadownet_2018-05-21-18-30-49.ckpt-6000'
# snapshot_name = '/shadownet_2018-05-21-18-30-49.ckpt-8000'
# snapshot_name = '/shadownet_2018-05-21-18-30-49.ckpt-10000'
# snapshot_name = '/shadownet_2018-05-21-18-30-49.ckpt-11000'
# snapshot_name = '/shadownet_2018-05-21-18-30-49.ckpt-12000'
# snapshot_name = '/shadownet_2018-05-21-18-30-49.ckpt-13000'
# snapshot_name = '/shadownet_2018-05-21-18-30-49.ckpt-14000'
# snapshot_name = '/shadownet_2018-05-21-18-30-49.ckpt-15000'
# snapshot_name = '/shadownet_2018-05-21-18-30-49.ckpt-16000'
# snapshot_name = '/shadownet_2018-05-22-12-41-07.ckpt-12000'
# snapshot_name = '/shadownet_2018-05-22-12-41-07.ckpt-13000'
snapshot_name = '/shadownet_2018-05-22-12-41-07.ckpt-14000'
snapshot_path = '/home/weiying1/hyg/icpr/CRNN_Tensorflow/model/shadownet-im-wchn-3'

# snapshot_name = '/shadownet_2018-05-19-19-17-51.ckpt-40000'
# snapshot_path = '/home/weiying1/hyg/icpr/CRNN_Tensorflow/model/shadownet-pre-chn'

beam_width = 10
# num_classes = 1800
num_classes = 4425#2292
# test_records = 'test_feature.tfrecords'
test_records = 'validation_feature.tfrecords'
tfrecord_dir = '/home/weiying1/hyg/data/icpr/tfrecords-f'
# tfrecord_dir = '/home/weiying1/hyg/data/icpr/tfrecords-gech'
model_test_result = '/home/weiying1/hyg/icpr/CRNN_Tensorflow/model/accuracy-im-wchn-3'
snapshot_path += snapshot_name
image_width = 800


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Where you store the test tfrecords data',
                        default=tfrecord_dir)
    parser.add_argument('--weights_path', type=str, help='Where you store the shadow net weights',
                        default=snapshot_path)
    parser.add_argument('--is_recursive', type=bool, help='If need to recursively test the dataset')

    return parser.parse_args()


def test_shadownet(dataset_dir, weights_path, is_vis=True, is_recursive=True):
    print('weights_path:', weights_path)
    print('beam_width:', beam_width)
    # Initialize the record decoder
    decoder = data_utils.TextFeatureIO().reader
    images_t, labels_t, widths_t, imagenames_t = decoder.read_features(
        ops.join(dataset_dir, test_records), num_epochs=None, image_width=image_width)
    if not is_recursive:
        images_sh, labels_sh, imagenames_sh = tf.train.shuffle_batch(tensors=[images_t, labels_t, imagenames_t],
                                                                     batch_size=32, capacity=1000+32*2,
                                                                     min_after_dequeue=2, num_threads=4)
    else:
        images_sh, labels_sh, widths_sh, imagenames_sh = tf.train.batch(tensors=[images_t, labels_t, widths_t, imagenames_t],
                                                             batch_size=32, capacity=1000 + 32 * 2, num_threads=4)
    images_sh = tf.cast(x=images_sh, dtype=tf.float32)
    input_widths = tf.squeeze(widths_sh, axis=1)

    # build shadownet
    net = crnn_model_im.ShadowNet(
        phase='Test', hidden_nums=256, layers_nums=2, seq_length=input_widths, num_classes=num_classes)
    with tf.variable_scope('shadow'):
        net_out = net.build_shadownet(inputdata=images_sh)
    decoded, _ = tf.nn.ctc_beam_search_decoder(net_out, input_widths, beam_width=beam_width, merge_repeated=False)
    seq_dist_mean = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels_sh))

    # config tf session
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH

    # config tf saver
    saver = tf.train.Saver()

    sess = tf.Session(config=sess_config)

    test_sample_count = 0
    for record in tf.python_io.tf_record_iterator(ops.join(dataset_dir, test_records)):
        test_sample_count += 1
    loops_nums = int(math.ceil(test_sample_count / 32))
    # loops_nums = 100

    with sess.as_default():

        # restore the model weights
        saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        print('Start predicting ......')
        if not is_recursive:
            predictions, images, labels, imagenames = sess.run([decoded, images_sh, labels_sh, imagenames_sh])
            imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
            imagenames = [tmp.decode('utf-8') for tmp in imagenames]
            preds_res = decoder.sparse_tensor_to_str(predictions[0])
            gt_res = decoder.sparse_tensor_to_str(labels)

            accuracy = []

            for index, gt_label in enumerate(gt_res):
                pred = preds_res[index]
                totol_count = len(gt_label)
                correct_count = 0
                try:
                    for i, tmp in enumerate(gt_label):
                        if tmp == pred[i]:
                            correct_count += 1
                except IndexError:
                    continue
                finally:
                    try:
                        accuracy.append(correct_count / totol_count)
                    except ZeroDivisionError:
                        if len(pred) == 0:
                            accuracy.append(1)
                        else:
                            accuracy.append(0)

            accuracy = np.mean(np.array(accuracy).astype(np.float32), axis=0)
            print('Mean test accuracy is {:5f}'.format(accuracy))

            for index, image in enumerate(images):
                print('Predict {:s} image with gt label: {:s} **** predict label: {:s}'.format(
                    imagenames[index], gt_res[index], preds_res[index]))
                if is_vis:
                    plt.imshow(image[:, :, (2, 1, 0)])
                    plt.show()
        else:
            accuracy = []
            seq_dist_mean_list = []
            # seq_dist_num_list = []
            for epoch in range(loops_nums):
                predictions, images, labels, imagenames = sess.run([decoded, images_sh, labels_sh, imagenames_sh])
                imagenames = np.reshape(imagenames, newshape=imagenames.shape[0])
                imagenames = [tmp.decode('utf-8') for tmp in imagenames]
                preds_res = decoder.sparse_tensor_to_str(predictions[0])
                gt_res = decoder.sparse_tensor_to_str(labels)
                acc_batch = []

                for index, gt_label in enumerate(gt_res):
                    pred = preds_res[index]
                    totol_count = len(gt_label)
                    correct_count = 0
                    acc = 0
                    try:
                        for i, tmp in enumerate(gt_label):
                            if tmp == pred[i]:
                                correct_count += 1
                    except IndexError:
                        print('IndexError, gt_label: ', gt_label, ' pred: ', pred)
                        continue
                    finally:
                        try:
                            acc = correct_count/totol_count
                        except ZeroDivisionError:
                            if len(pred) == 0:
                                acc = 1
                            else:
                                acc = 0
                        accuracy.append(acc)
                        acc_batch.append(acc)
                    print('Predict {:s} image with gt label: {:s} **** predict label: {:s}'.format(
                        imagenames[index], gt_res[index], preds_res[index]))

                seq_dist_mean_v = sess.run(seq_dist_mean)
                # seq_dist_num_v = sess.run(seq_dist_num)
                seq_dist_mean_list.append(seq_dist_mean_v)
                # seq_dist_num_list.append(seq_dist_num_v)
                print('Batch count: %d, batch index: %d.' % (loops_nums, epoch+1))
                print('This batch accuracy_mean = ', np.mean(np.array(acc_batch).astype(np.float32)))
                print('This batch seq_dist_mean = ', seq_dist_mean_v)
                # print('seq_dist_num = ', seq_dist_num_v)

                # for index, image in enumerate(images):
                #     if is_vis:
                #         plt.imshow(image[:, :, (2, 1, 0)])
                #         plt.show()

            accuracy_all = np.mean(np.array(accuracy).astype(np.float32), axis=0)
            seq_dist_all_mean = np.mean(np.array(seq_dist_mean_list).astype(np.float32))
            # seq_dist_num_all = np.sum(np.array(seq_dist_num_list), axis=0)
            # seq_dist_num_all_mean = np.mean(np.array(seq_dist_num_list).astype(np.float32), axis=0)/32
            print('Test accuracy_all is {:5f}'.format(accuracy_all))
            print('Test seq_dist_all_mean is {:5f}'.format(seq_dist_all_mean))
            # print('Test seq_dist_num_all is {:d}'.format(seq_dist_num_all))
            # print('Test seq_dist_num_all_mean is {:5f}'.format(seq_dist_num_all_mean))
            with open(model_test_result, 'a', encoding='utf-8') as f:
                f.write(snapshot_name[1:]+':\n')
                f.write('Test accuracy_all is {}'.format(accuracy_all)+'\n')
                f.write('Test seq_dist_all_mean is {}'.format(seq_dist_all_mean)+'\n\n')

        coord.request_stop()
        coord.join(threads=threads)

    sess.close()
    return


if __name__ == '__main__':
    # init args
    args = init_args()
    
#     dataset_dir = 'data'
#     weights_path = 'model/shadownet/shadownet_2017-10-17-11-47-46.ckpt-199999'
    
    # test shadow net
    test_shadownet(args.dataset_dir, args.weights_path, args.is_recursive)
#     test_shadownet(dataset_dir, weights_path)


