# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2023/2/28 8:36
# @Author : Zhu Shujiang
# @Email : 695699484@qq.com
# @File : mnist_train.py
# @Project : tensorflow_cpu
# @Description: mnist训练模型及保存模块

import tensorflow as tf
from config import init_config
import mnist_inference
from tensorflow_core.examples.tutorials.mnist import input_data
config = init_config.Config.config['config']
SAVE_MODEL_PB = config['model_save_path']['pb']
SAVE_MODEL_CKPT = config['model_save_path']['ckpt']
network_param = config['network_param']
BATCH_SIZE = network_param['batch_size']
TRAINING_STEPS = network_param['training_steps']
LEARNING_RATE_BASE = network_param['learning_rate_base']
LEARNING_RATE_DECAY = network_param['learning_rate_decay']
REGULARIZATION_RATE = network_param['regularization_rate']
MOVING_AVERAGE_DECAY = network_param['moving_average_decay']
data_path = config['data']


def train(mnist):
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x_input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y_input')
    # 定义正则化函数
    regularization = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = mnist_inference.inference(x, regularization)
    # 定义一个记录训练轮数的变量，不可被训练
    global_step = tf.Variable(0, trainable=False)
    # 定义滑动平均影子变量
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_step = variable_averages.apply(tf.trainable_variables())
    # 定义损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_op = tf.group(train_step, variable_averages_step, name='train')

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 进行训练
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                print("After {} training step(s), loss on training "
                      "batch is {}.".format(step, loss_value))
                saver.save(sess, SAVE_MODEL_CKPT, global_step=step)
        graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['train'])
        with tf.gfile.GFile(SAVE_MODEL_PB, 'wb') as f:
            f.write(output_graph_def.SerializeToString())


def main(argv=None):
    mnist = input_data.read_data_sets(data_path, one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()



