# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2023/2/28 8:36
# @Author : Zhu Shujiang
# @Email : 695699484@qq.com
# @File : mnist_eval.py
# @Project : tensorflow_cpu
# @Description: 验证、测试
import os
import time
from config import init_config
import tensorflow as tf
import mnist_inference
import mnist_train
from tensorflow_core.examples.tutorials.mnist import input_data
config = init_config.Config.config['config']

# 每 10 秒加载一次最新的模型，并在测试数据上测试最新模型的正确率。
EVAL_INTERVAL_SECS = 10
data_path = config['data']


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x_input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y_input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        y = mnist_inference.inference(x)  # 定义前向传播结果，此处为未使用滑动平均变量
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重名名的方式来加载模型，就不用在前向传播过程中调用滑动平均函数来获取平均值,加载完成后，上面的y就变成了计算了滑动平均变量的y
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(os.path.dirname(mnist_train.SAVE_MODEL_CKPT))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argV=None):
    mnist = input_data.read_data_sets(data_path, one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()