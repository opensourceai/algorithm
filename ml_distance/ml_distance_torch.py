# -*- coding: UTF-8 -*-
# File Name：ml_distance
# Author : Chen Quan
# Date：2019/2/18
# Description : Use Torch to implement a distance learning formula for machine learning.
__author__ = 'Chen Quan'
import torch


# 欧式距离
def euclidean_distance(a, b):
    """
    n dim
    :param a:
    :param b:
    :return:
    """
    a, b = a.float(), b.float()
    return torch.sqrt(torch.sum(torch.pow(a - b, 2)))


# 标准欧式距离
def standard_euclidean_distance(a, b):
    """

    :param a:
    :param b:
    :return:
    """
    a, b = a.float(), b.float()

    avg = (a + b) / 2
    s = torch.pow((a - avg), 2) + torch.pow((b - avg), 2)
    d = torch.sum(torch.pow(a - b, 2) / s)
    return d


# 曼哈顿距离
def manhattan_distance(a, b):
    a, b = a.float(), b.float()

    return torch.sum(torch.abs(a - b))


# 切比雪夫距离
def chebyshev_distance(a, b):
    a, b = a.float(), b.float()
    return torch.max(torch.abs(a - b))


# 夹角余弦距离
def cos_distance(a, b):
    a, b = a.float(), b.float()

    molecule = torch.sum(a * b)
    denominator = torch.sqrt(torch.sum(torch.pow(a, 2)) + torch.sum(torch.pow(b, 2)))
    return molecule / denominator


# 汉明距离
def hamming_distance(a, b):
    a, b = a.float(), b.float()

    return torch.sum(a != b)


# 杰卡德系数
def Jaccard_similarity_coefficient(a, b):
    """
    暂无法实现
    :param a:
    :param b:
    :return:
    """
    pass
