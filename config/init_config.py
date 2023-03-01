# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2022/6/10 15:18
# @Author : Zhu Shujiang
# @Email : 695699484@qq.com
# @File : init_config.py
# @Project : gongKuangProject
# @Description: 读取配置

import sys
import os
import threading
import yaml
import time


def read_init_config():
    config = dict()
    config["root_dir"] = sys.path[0]
    config["project_dir"] = os.path.dirname(config["root_dir"])
    config["config_dir"] = os.path.join(config["root_dir"], "config")
    config["config_file"] = os.path.join(config["config_dir"], "config.yaml")

    with open(config["config_file"], "r", encoding="utf-8") as file:
        config["config"] = yaml.safe_load(file)

    config["config"]["log"]["logdir"] = os.path.join(config["root_dir"], config["config"]["log"]["logdir"])
    return config


class Config(object):
    _instance_lock = threading.Lock()
    config = read_init_config()
    flag = False

    def __init__(self):
        if Config.flag:
            return
        Config.flag = True
        pass

    def __new__(cls, *args, **kwargs):
        if not hasattr(Config, "_instance"):
            with Config._instance_lock:
                if not hasattr(Config, "_instance"):
                    Config._instance = object.__new__(cls)
        return Config._instance

