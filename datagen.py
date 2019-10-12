"""
SRNet - Editing Text in the Wild
Data generator.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import os
import cv2
import numpy as np
import random
import cfg 

def srnet_datagen():
   
    # generator SRNet data for training
    name_list = os.listdir(os.path.join(cfg.data_dir, cfg.t_b_dir))
    random.shuffle(name_list)
    name_num = len(name_list)
    idx = 0
    
    while True:
        i_t_batch, i_s_batch = [], []
        t_sk_batch, t_t_batch, t_b_batch, t_f_batch = [], [], [], []
        mask_t_batch = []
        
        for _ in range(cfg.batch_size):
            name = name_list[idx]

            i_t = cv2.imread(os.path.join(cfg.data_dir, cfg.i_t_dir, name))
            i_s = cv2.imread(os.path.join(cfg.data_dir, cfg.i_s_dir, name))
            t_sk = cv2.imread(os.path.join(cfg.data_dir, cfg.t_sk_dir, name), cv2.IMREAD_GRAYSCALE)
            t_t = cv2.imread(os.path.join(cfg.data_dir, cfg.t_t_dir, name))
            t_b = cv2.imread(os.path.join(cfg.data_dir, cfg.t_b_dir, name))
            t_f = cv2.imread(os.path.join(cfg.data_dir, cfg.t_f_dir, name))
            mask_t = cv2.imread(os.path.join(cfg.data_dir, cfg.mask_t_dir, name), cv2.IMREAD_GRAYSCALE)

            i_t_batch.append(i_t)
            i_s_batch.append(i_s)
            t_sk_batch.append(t_sk)
            t_t_batch.append(t_t)
            t_b_batch.append(t_b)
            t_f_batch.append(t_f)
            mask_t_batch.append(mask_t)
            idx = (idx + 1) % name_num
        
        w_sum = 0
        for t_b in t_b_batch:
            h, w = t_b.shape[:2]
            scale_ratio = cfg.data_shape[0] / h
            w_sum += int(w * scale_ratio)
        
        to_h = cfg.data_shape[0]
        to_w = w_sum // cfg.batch_size
        to_w = int(round(to_w / 8)) * 8
        to_scale = (to_w, to_h) # w first for cv2
        for i in range(cfg.batch_size): 
            i_t_batch[i] = cv2.resize(i_t_batch[i], to_scale)
            i_s_batch[i] = cv2.resize(i_s_batch[i], to_scale)
            t_sk_batch[i] = cv2.resize(t_sk_batch[i], to_scale, interpolation=cv2.INTER_NEAREST)
            t_t_batch[i] = cv2.resize(t_t_batch[i], to_scale)
            t_b_batch[i] = cv2.resize(t_b_batch[i], to_scale)
            t_f_batch[i] = cv2.resize(t_f_batch[i], to_scale)
            mask_t_batch[i] = cv2.resize(mask_t_batch[i], to_scale, interpolation=cv2.INTER_NEAREST)

        i_t_batch = np.stack(i_t_batch)
        i_s_batch = np.stack(i_s_batch)
        t_sk_batch = np.expand_dims(np.stack(t_sk_batch), axis = -1)
        t_t_batch = np.stack(t_t_batch)
        t_b_batch = np.stack(t_b_batch)
        t_f_batch = np.stack(t_f_batch)
        mask_t_batch = np.expand_dims(np.stack(mask_t_batch), axis = -1)
        
        i_t_batch = i_t_batch.astype(np.float32) / 127.5 - 1. 
        i_s_batch = i_s_batch.astype(np.float32) / 127.5 - 1. 
        t_sk_batch = t_sk_batch.astype(np.float32) / 255. 
        t_t_batch = t_t_batch.astype(np.float32) / 127.5 - 1. 
        t_b_batch = t_b_batch.astype(np.float32) / 127.5 - 1. 
        t_f_batch = t_f_batch.astype(np.float32) / 127.5 - 1. 
        mask_t_batch = mask_t_batch.astype(np.float32) / 255.
        
        yield [i_t_batch, i_s_batch, t_sk_batch, t_t_batch, t_b_batch, t_f_batch, mask_t_batch]

def get_input_data(data_dir = cfg.example_data_dir):
    
    # get input data from dir
    data_list = os.listdir(data_dir)
    data_list = [data_name.split('_')[0] + '_' for data_name in data_list]
    data_list = list(set(data_list))
    res_list = []
    for data_name in data_list:
        i_t = cv2.imread(os.path.join(cfg.example_data_dir, data_name + 'i_t.png'))
        i_s = cv2.imread(os.path.join(cfg.example_data_dir, data_name + 'i_s.png'))
        h, w = i_t.shape[:2]
        scale_ratio = cfg.data_shape[0] / h
        to_h = cfg.data_shape[0]
        to_w = int(round(int(w * scale_ratio) / 8)) * 8
        to_scale = (to_w, to_h) # w first for cv2
        i_t = cv2.resize(i_t, to_scale).astype(np.float32) / 127.5 - 1.
        i_s = cv2.resize(i_s, to_scale).astype(np.float32) / 127.5 - 1.
        i_t = np.expand_dims(i_t, axis = 0)
        i_s = np.expand_dims(i_s, axis = 0)
        res_list.append([i_t, i_s, (w, h), data_name]) # w first for cv2
    return res_list
