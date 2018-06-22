#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-9-22 afternoon 1:39
# @Author  : Luo Yao
# @Site    : http://github.com/TJCVRS
# @File    : train_shadownet.py
# @IDE: PyCharm Community Edition
"""
Train shadow net script
"""
import os, sys
import tensorflow as tf
import os.path as ops
import time
import numpy as np
import argparse
root = '/home/weiying1/hyg/icpr/CRNN_Tensorflow'
os.chdir(root)
sys.path.insert(0, root)
from crnn_model import crnn_model_im
from local_utils import data_utils_im, log_utils
from global_configuration import config_im


config = config_im
data_utils = data_utils_im
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
beam_width = 10
num_classes = 4425
# num_classes = 1800
# num_classes = 2292
snapshot_path = '/home/weiying1/hyg/icpr/CRNN_Tensorflow/model/shadownet-pre-chn'
tfboard_save_path = 'tboard/shadownet-pre-chn'
model_save_dir = 'model/shadownet-pre-chn'
tfrecord_dir = '/home/weiying1/hyg/data/icpr/tfrecords-gech'
# snapshot_path += '/shadownet_2018-05-17-16-32-28.ckpt-520000'
snapshot_path += '/shadownet_2018-05-19-09-21-27.ckpt-20000'
# snapshot_path = None
start_iter = 20000
image_width = 48
logger = log_utils.init_logger()


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, help='Where you store the dataset',
                        default=tfrecord_dir)
    parser.add_argument('--weights_path', type=str, help='Where you store the pretrained weights',
                        default=snapshot_path)

    return parser.parse_args()


def train_shadownet(dataset_dir, batch_size, seq_length, weights_path=None):
    """

    :param dataset_dir:
    :param weights_path:
    :return:
    """
    
    # decode the tf records to get the training data
    decoder = data_utils.TextFeatureIO().reader
    images, labels, widths, imagenames = decoder.read_features(
        ops.join(dataset_dir, 'train_feature.tfrecords'), num_epochs=None, image_width=image_width)
    inputdata, input_labels, input_widths, input_imagenames = tf.train.shuffle_batch(
        tensors=[images, labels, widths, imagenames],
        batch_size=batch_size,
        capacity=57600,#2000,#
        min_after_dequeue=48000,#1600,#
        num_threads=1)
    inputdata = tf.cast(x=inputdata, dtype=tf.float32)
    input_widths = tf.squeeze(input_widths, axis=1)
    # input_widths = tf.div(input_widths, 4)

    # initializa the net model
    shadownet = crnn_model_im.ShadowNet(phase='Train', hidden_nums=256, layers_nums=2,
                                     seq_length=input_widths, num_classes=num_classes)
    with tf.variable_scope('shadow', reuse=False):
        net_out = shadownet.build_shadownet(inputdata=inputdata)
    cost = tf.reduce_mean(tf.nn.ctc_loss(labels=input_labels,
                                         inputs=net_out,
                                         sequence_length=input_widths,
                                         ignore_longer_outputs_than_inputs=True))
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(net_out, input_widths, beam_width=beam_width, merge_repeated=False)
    sequence_dist = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), input_labels))

    global_step = tf.Variable(0, name='global_step', trainable=False)
    # global_step1 = tf.assign(global_step, 0)
    starter_learning_rate = config.cfg.TRAIN.LEARNING_RATE
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               config.cfg.TRAIN.LR_DECAY_STEPS, config.cfg.TRAIN.LR_DECAY_RATE,
                                               staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(loss=cost, global_step=global_step)

    # Set tf summary
    if not ops.exists(tfboard_save_path):
        os.makedirs(tfboard_save_path)
    
    tf.summary.scalar(name='Cost', tensor=cost)
    tf.summary.scalar(name='Learning_Rate', tensor=learning_rate)
    tf.summary.scalar(name='Seq_Dist', tensor=sequence_dist)
    merge_summary_op = tf.summary.merge_all()
    accuracy_holder = tf.placeholder(tf.float32)
    summary_accuracy = tf.summary.scalar(name='Accuracy', tensor=accuracy_holder)
    
    # Set saver configuration
    saver = tf.train.Saver(max_to_keep=20)
    if not ops.exists(model_save_dir):
        os.makedirs(model_save_dir)
    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'shadownet_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = ops.join(model_save_dir, model_name)

    # Set sess configuration
#     sess_config = tf.ConfigProto(device_count = {'GPU': 0})
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = config.cfg.TRAIN.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = config.cfg.TRAIN.TF_ALLOW_GROWTH
    
    sess = tf.Session(config=sess_config)
    
    summary_writer = tf.summary.FileWriter(tfboard_save_path)
    summary_writer.add_graph(sess.graph)

    # Set the training parameters
    train_epochs = config.cfg.TRAIN.EPOCHS

    with sess.as_default():
        if weights_path is None:
            logger.info('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            logger.info('Restore model from {:s}'.format(weights_path))
            saver.restore(sess=sess, save_path=weights_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        
        for epoch in range(start_iter, train_epochs+1):
            if epoch % config.cfg.TRAIN.DISPLAY_STEP == 0:
                logger.info('Epoch: {:d} global_step: {:d}'.format(epoch, sess.run(global_step)))
            
            # if epoch % 5000 == 0 and epoch > 50000:
            if epoch % 500 == 0:
                c, seq_distance, preds, gt_labels, summary = sess.run(
                [cost, sequence_dist, decoded, input_labels, merge_summary_op])
                
                # calculate the precision
                preds = decoder.sparse_tensor_to_str(preds[0])
                gt_labels = decoder.sparse_tensor_to_str(gt_labels)
                
                accuracy = []
                
                for index, gt_label in enumerate(gt_labels):
                    pred = preds[index]
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
                
                logger.info('Epoch: {:d} cost= {:9f} seq distance= {:9f} train accuracy= {:9f}'.format(
                    epoch, c, seq_distance, accuracy))
                
                summary_writer.add_summary(summary=summary, global_step=epoch)
                sa = sess.run(summary_accuracy, feed_dict={accuracy_holder: accuracy})
                summary_writer.add_summary(summary=sa, global_step=epoch)

            # if epoch % 20000 == 0 and epoch > 50000:
            if epoch % 5000 == 0:
                saver.save(sess=sess, save_path=model_save_path, global_step=epoch)
            sess.run(optimizer)
            
        coord.request_stop()
        coord.join(threads=threads)
        
    sess.close()
    
    return


if __name__ == '__main__':
    # init args
    args = init_args()
    
    if not ops.exists(args.dataset_dir):
        raise ValueError('{:s} doesn\'t exist'.format(args.dataset_dir))
    
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     tf.reset_default_graph()
    train_shadownet(args.dataset_dir, batch_size=30*32, seq_length=50, weights_path=args.weights_path)
    print('Done')
    
    
    
    
    
    
    
    
