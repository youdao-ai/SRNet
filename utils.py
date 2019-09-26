"""
SRNet - Editing Text in the Wild
Common utility functions and classes.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
import os
import cv2
from datetime import datetime

PrintColor = {
    'black': 30,
    'red': 31,
    'green': 32,
    'yellow': 33,
    'blue': 34,
    'amaranth': 35,
    'ultramarine': 36,
    'white': 37
}

PrintStyle = {
    'default': 0,
    'highlight': 1,
    'underline': 4,
    'flicker': 5,
    'inverse': 7,
    'invisible': 8
}

def get_train_name():
    
    # get current time for train name
    return datetime.now().strftime('%Y%m%d%H%M%S')

def print_log(s, time_style = PrintStyle['default'], time_color = PrintColor['blue'],
                content_style = PrintStyle['default'], content_color = PrintColor['white']):
    
    # colorful print s with time log
    cur_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    log = '\033[{};{}m[{}]\033[0m \033[{};{}m{}\033[0m'.format \
        (time_style, time_color, cur_time, content_style, content_color, s)
    print (log)
    
def print_nodes(graph):
    
    # print all nodes of the graph
    nodes = [n.name for n in graph.as_graph_def().node]
    for node in nodes:
        print (node)

def write_summary(d_writer, g_writer, d_log, g_log, global_step):

    # write summaries for tensorboard
    d_writer.add_summary(d_log, global_step)
    g_writer.add_summary(g_log, global_step)

def save_result(save_dir, result, name, mode):

    # save output images
    o_sk, o_t, o_b, o_f = result
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(os.path.join(save_dir, name + 'o_f.png'), o_f, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    if mode == 1:
        cv2.imwrite(os.path.join(save_dir, name + 'o_sk.png'), o_sk, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(os.path.join(save_dir, name + 'o_t.png'), o_t, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        cv2.imwrite(os.path.join(save_dir, name + 'o_b.png'), o_b, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

def predict_data_list(model, sess, save_dir, input_data_list, mode = 1):

    # predict output images and save them
    for data in input_data_list:
        i_t, i_s, ori_shape, data_name = data
        result = model.predict(sess, i_t, i_s, ori_shape)
        save_result(save_dir, result, data_name, mode = mode)

def save_checkpoint(sess, saver, save_dir, global_step):
    
    # save tensorflow ckpt files
    saver.save(sess, save_dir, global_step = global_step)

def save_pb(sess, save_path, outputs = ['o_sk', 'o_t', 'o_b', 'o_f']):

    # save tensorflow pb model
    save_dir = os.path.split(save_path)[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, outputs)
    with tf.gfile.FastGFile(save_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())

