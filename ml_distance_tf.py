# -*- coding: UTF-8 -*-
# File Name：ml_distance
# Author : Chen Quan
# Date：2019/2/18
# Description :
__author__ = 'Chen Quan'
import tensorflow as tf

layers = tf.keras.layers


# 欧式距离
def euclidean_distance(a, b):
    """
    n dim
    :param a:
    :param b:
    :return:
    """
    return tf.sqrt(tf.reduce_sum(tf.square(a - b, name='square')), name='euclidean_distance')


# 标准欧式距离
def standard_euclidean_distance(a, b):
    """

    :param a:
    :param b:
    :return:
    """
    avg = (a + b) / 2
    s = tf.square((a - avg)) + tf.square((b - avg))
    d = tf.reduce_sum(tf.square(a - b) / s)
    return d


# 曼哈顿距离
def manhattan_distance(a, b):
    a, b = tf.cast((a, b), dtype=tf.float32)
    return tf.reduce_sum(tf.abs(a - b))


# 切比雪夫距离
def chebyshev_distance(a, b):
    a, b = tf.cast((a, b), dtype=tf.float32)
    return tf.reduce_max(tf.abs(a - b))


# 夹角余弦距离
def cos_distance(a, b):
    a, b = tf.cast((a, b), dtype=tf.float32)
    molecule = tf.reduce_sum(a * b)
    denominator = tf.sqrt(tf.reduce_sum(tf.square(a)) + tf.reduce_sum(tf.square(b)))
    return molecule / denominator


# 汉明距离
def hamming_distance(a, b):
    a, b = tf.cast((a, b), dtype=tf.float32)
    cast = tf.cast(tf.not_equal(a, b), tf.float32)
    return tf.reduce_sum(cast)


# 杰卡德系数
def Jaccard_similarity_coefficient(a, b):
    pass
