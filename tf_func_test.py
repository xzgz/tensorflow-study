#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018-04-24
# @Author  : Yangguang He
# @Site    : 
# @Brief   : Test the usage of some tensorflow and numpy functions.
# @File    : tf_func_test.py
# @IDE: CLion
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.reset_default_graph()

# X = tf.Variable(tf.truncated_normal([100], stddev=0.1, name="X"))
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# X = sess.run(X)
# print(X)

def test_truncated_normal():
    num = 1000000
    X = tf.truncated_normal([num], stddev=0.1, name="X")
    sess = tf.Session()
    X = sess.run(X)
    # print(np.int32([0.99, -0.99, -1.0, -1.1, -1.99, 1.0, 1.1, 1.99, 2.0]))
    print(100/num, type(100/num), num/100, type(num/100))
    keys = np.arange(-0.2, 0.201, 100/num)

    temp = {}
    for v in X:
        index = np.int32(num/100 * v + 2 * num/1000)
        if index not in temp:
            temp[index] = 0
        else:
            temp[index] += 1

    data_x = []
    data_y = []
    normal_cnt = 0
    i = 0
    for k in keys:
        if i in temp:
            data_x.append(k)
            data_y.append(temp[i])
            normal_cnt += temp[i]
        i += 1
    print('num-normal_cnt:', num-normal_cnt)
    plt.plot(data_x, data_y)
    plt.show()
# test_truncated_normal()


def test_variable_assign():
    print('Method 1:')
    x = tf.Variable(0)
    y = tf.assign(x, 1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('before assign:', sess.run(x))
        print('y after assign:', sess.run(y))
        print('x after assign:', sess.run(x))
    # sess.close()

    print('\nMethod 2:')
    w = tf.Variable(2)
    w_new = w.assign(3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('before assign:', sess.run(w))
        # print('after assign:', sess.run(w_new))
        print('after assign:', w_new.eval())
    # sess.close()

    print('\nMethod 3:')
    z = tf.Variable(7)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print('before assign:', sess.run(z))
    z.load(8, sess)
    print('after assign:', sess.run(z))
    # sess.close()
# test_variable_assign()


def test_save_restore_model():
    v1 = tf.Variable(7, name="v1")
    v2 = tf.Variable(8, name="v2")
    # v1 = v1.assign(20)
    v1_n = tf.assign(v1, 20)
    v2 = tf.assign(v2, 30)
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        print('v1 before restore:', sess.run(v1))
        print('v2 before restore:', sess.run(v2))
        print('v1_n before restore:', sess.run(v1_n))
        # saver.save(sess, "test_save_restore_model/model.ckpt")
        saver.restore(sess, "test_save_restore_model/model.ckpt")
        print('v1 after restore:', sess.run(v1))
        print('v2 after restore:', sess.run(v2))
        print('v1_n after restore:', sess.run(v1_n))
# test_save_restore_model()


def test_edit_distance():
    # (0, 0) = ["a"]
    # (1, 0) = ["b"]
    hypothesis = tf.SparseTensor(
        [[0, 0, 0],
         [1, 0, 0]],
        ["a", "b"],
        (2, 1, 1))
    # (0, 0) = ["a"]
    # (0, 1) = []
    # (1, 0) = ["b", "c"]
    # (1, 1) = ["a"]
    truth = tf.SparseTensor(
        [[0, 0, 0],
         [1, 0, 0],
         [1, 0, 1],
         [1, 1, 0]],
        ["a", "b", "c", "a"],
        (2, 2, 2))
    output = tf.edit_distance(hypothesis, truth, normalize=False)
    with tf.Session() as sess:
        output = sess.run(output)
        # print(output.eval())
        print(output, type(output), output.dtype)
# test_edit_distance()


def test_squeeze():
    X1 = np.random.randn(1, 4, 5)
    X2 = np.random.randn(1, 4, 5)
    X2[0, 2:] = 0
    print(type(X1), X1.shape)
    X = np.array([X1, X2])
    print(type(X), X.shape)
    X = tf.squeeze(X, axis=1)
    print(type(X), X.shape)

    print('X1:\n', X1)
    print('X2:\n', X2)

    sess = tf.Session()
    print(X)
    print('X:\n', sess.run(X))
# test_squeeze()


def test_pad():
    # t = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int64)
    t = tf.constant([[1, 2, 3], [4, 5, 6]], tf.float32)
    paddings = tf.constant([[0, 2], [0, 2]])
    # c = tf.constant([[4, 6], [1, 2]])
    # c = tf.constant(4, dtype=tf.int64)
    c = tf.constant(0, tf.float32)
    print(t.shape, t.dtype)
    print(paddings.shape, paddings.dtype)
    print(c.dtype)
    tc = tf.pad(t, paddings, "CONSTANT", constant_values=c)
    tr = tf.pad(t, paddings, "REFLECT")
    ts = tf.pad(t, paddings, "SYMMETRIC")
    t1 = tf.pad(t, [[0, 1], [0, 2]], "CONSTANT", constant_values=c)
    t2 = tf.pad(t, [[0, 1], [0, 4]], "CONSTANT", constant_values=c)
    t3 = tf.concat([t1, t2], axis=1)
    vec1 = tf.constant([1, 2, 4.0])
    # vec1 = tf.constant([1, 2, 4])
    vec2 = vec1+3
    # vec3 = vec1 / vec2
    # vec3 = tf.divide(vec1, vec2)
    vec3 = tf.div(vec1, vec2)
    print(vec1.dtype, vec2.dtype)
    sess = tf.Session()

    # print('tc:\n', sess.run(tc))
    # print('tr:\n', sess.run(tr))
    # print('ts:\n', sess.run(ts))
    print('tc:\n', sess.run(t1))
    print('tc:\n', sess.run(t2))
    print('tc:\n', sess.run(t3))
    print('vec1:\n', sess.run(vec1))
    print('vec2:\n', sess.run(vec2))
    print('vec3:\n', sess.run(vec3))

    sess.close()

    # a = [[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]]
    # # ac = np.pad(a, [[0, 2], [0, 1]], 'constant', constant_values=[[4, 6], [1, 2]])
    # # ac = np.pad(a, [[0, 2]], 'constant', constant_values=[[4, 6], [1, 2]])
    # print('before pad shape of:', np.array(a).shape)
    # ac = np.pad(a, [[0, 0], [0, 3], [0, 0]], 'constant', constant_values=[[1, 6], [2, 0], [4, 0]])
    # print('after pad shape of ac:', ac.shape)
    # ar = np.pad(a, [[1, 0]], 'reflect')
    # as1 = np.pad(a, [[1, 0]], 'symmetric')
    # print('ac:\n', ac)
    # print('ar:\n', ar)
    # print('as1:\n', as1)
# test_pad()


def test_tensor_shape():
    t1 = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
    t2 = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
    print('tf.shape(t1):', tf.shape(t1))
    print('tf.shape(t2):', tf.shape(t2))
    print('t1.shape:', t1.shape)
    print('t2.shape:', t2.shape)
    print('t1.get_shape():', t1.get_shape())
    print('t2.get_shape():', t2.get_shape())
    sess = tf.Session()

    print('run tf.shape(t1):', sess.run(tf.shape(t1)))
    # print(sess.run(tf.shape(t2)))
# test_tensor_shape()


def test_variable_scope():
    with tf.variable_scope("foo"):
        v = tf.get_variable("v", [1])
    with tf.variable_scope("foo", reuse=True):
        v1 = tf.get_variable("v", [1])
    print('v:', v.name)
    print('v1:', v1.name)
    assert v1 == v
# test_variable_scope()


def test_placeholder():
    x = tf.placeholder(tf.float32, shape=(1024, 1024))
    y = tf.matmul(x, x)
    print('x.dtype:', x.dtype)
    print('y.dtype:', y.dtype)

    with tf.Session() as sess:
        # print(sess.run(y))                              # ERROR: will fail because x was not fed.
        rand_array = np.random.rand(1024, 1024)
        print('x.dtype:', x.dtype)
        print('y.dtype:', y.dtype)
        print('rand_array.dtype:', rand_array.dtype)
        print(sess.run(y, feed_dict={x: rand_array}))   # Will succeed.
# test_placeholder()


def test_divide():
    # t1 = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
    # t2 = 1/t1
    # # t3 = tf.divide(1.0, t1)  # error
    # t4 = tf.div(1, t1)
    # print('t1 dtype:', t1.dtype)
    # print('t2 dtype:', t2.dtype)
    # # print('t3 dtype:', t3.dtype)
    # print('t4 dtype:', t4.dtype)
    # sess = tf.Session()
    # print('t1:\n', sess.run(t1))
    # print('t2:\n', sess.run(t2))
    # # print('t3:\n', sess.run(t3))
    # print('t4:\n', sess.run(t4))

    t = tf.constant([-22.00627136, -21.70117188, -10.46626282, -9.78709602, -9.5978241,
                     -6.49891472, -2.78729439, -14.46313477, 0.32265404, -2.64193416])
    print(t.dtype)
    index = tf.argmax(t, axis=0)
    sess = tf.Session()
    print('index:', sess.run(index))
test_divide()





