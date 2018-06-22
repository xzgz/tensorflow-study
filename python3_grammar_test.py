#!/usr/bin/env python

import time


def test_init_del_method_in_class():
    class Animal(object):

        # 初始化方法
        # 创建完对象后会自动被调用
        def __init__(self, name):
            print('__init__方法被调用')
            self.__name = name

        # 析构方法
        # 当对象被删除时，会自动被调用
        def __del__(self):
            print("__del__方法被调用")
            print("%s对象马上被干掉了..."%self.__name)
            del self.__name

    # 创建对象
    dog = Animal("哈皮狗")

    # 删除对象
    del dog

    cat = Animal("波斯猫")
    cat2 = cat
    cat3 = cat

    print("---马上 删除cat对象")
    del cat
    print("---马上 删除cat2对象")
    del cat2
    print("---马上 删除cat3对象")
    del cat3

    print("程序2秒钟后结束")
    time.sleep(2)
# test_init_del_method_in_class()


def test_call_method_in_class():
    class X(object):
        def __init__(self, a, b, range):
            self.a = a
            self.b = b
            self.range = range

        def __call__(self, a, b):
            self.a = a
            self.b = b
            print('__call__ with （{}, {}）'.format(self.a, self.b))

        def __del__(self):
            del self.a
            del self.b
            del self.range

        # def __del__(self, a, b, range):
        #     del self.a
        #     del self.b
        #     del self.range

    instance_x = X(1, 2, 3)
    instance_x(1, 2)
test_call_method_in_class()




