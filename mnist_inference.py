# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2023/2/28 8:35
# @Author : Zhu Shujiang
# @Email : 695699484@qq.com
# @File : mnist_inference.py
# @Project : tensorflow_cpu
# @Description: mnist前向传播

import tensorflow as tf
from config import init_config
config = init_config.Config.config["config"]
network_param = config['network_param']
INPUT_NODE = network_param['input_node']
OUTPUT_NODE = network_param['output_node']
LAYER1_NODE = network_param['layer1_node']


def get_weight_variable(shape, regularization=None):
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularization is not None:
        tf.add_to_collection('losses', regularization(weights))
    return weights


def inference(input_tensor, regularization=None):
    with tf.variable_scope('layer_1'):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularization)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope('layer_2'):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularization)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2

