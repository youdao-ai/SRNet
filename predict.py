"""
SRNet - Editing Text in the Wild
Data prediction.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import tensorflow as tf
from model import SRNet
import numpy as np
import os
import cfg
from utils import *
from data_gen import srnet_datagen, get_input_data
import argparse

def main():
    
    parser = argparse.ArgumentParser() 
    parser.add_argument('--gpu', help = 'gpu id', default = 0)
    parser.add_argument('--i_s', help = 'input original text patch')
    parser.add_argument('--i_t', help = 'input standard text patch')
    parser.add_argument('--input_dir', help = 'Directory containing xxx_i_s and xxx_i_t with same prefix',
                        default = cfg.predict_data_dir)
    parser.add_argument('--save_dir', help = 'Directory to save result', default = cfg.predict_result_dir)
    parser.add_argument('--save_mode', help = '1 to save all and 0 to save onle o_f', type = int, default = 0)
    parser.add_argument('--checkpoint', help = 'tensorflow ckpt', default = cfg.predict_ckpt_path)
    args = parser.parse_args()

    assert (args.input_dir is not None and args.i_s is None and args.i_t is None) \
            or (args.input_dir is None and args.i_s is not None and args.i_t is not None)
    assert args.save_dir is not None
    assert args.save_mode == 0 or args.save_mode == 1
    assert args.checkpoint is not None

    # gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # define model
    print_log('model compiling start.', content_color = PrintColor['yellow'])
    model = SRNet(shape = cfg.data_shape, name = 'predict')
    print_log('model compiled.', content_color = PrintColor['yellow'])

    with model.graph.as_default():
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            
            # load pretrained weights
            print_log('weight loading start.', content_color = PrintColor['yellow'])
            saver.restore(sess, args.checkpoint)
            print_log('weight loaded.', content_color = PrintColor['yellow'])
            
            # predict
            print_log('predicting start.', content_color = PrintColor['yellow'])
            if args.input_dir is None:
                i_s = cv2.imread(args.i_s)
                i_t = cv2.imread(args.i_t)
                o_sk, o_t, o_b, o_f = model.predict(sess, i_t, i_s)
                
                cv2.imwrite(os.path.join(args.save_dir, 'result.png'), o_f)
                if args.save_mode == 1:
                    cv2.imwrite(os.path.join(args.save_dir, 'result_sk.png'), o_sk)
                    cv2.imwrite(os.path.join(args.save_dir, 'result_t.png'), o_t)
                    cv2.imwrite(os.path.join(args.save_dir, 'result_b.png'), o_b)
            else:
                predict_data_list(model, sess, args.save_dir, get_input_data(args.input_dir), mode = args.save_mode)
            print_log('predicting finished.', content_color = PrintColor['yellow'])
            
if __name__ == '__main__':
    main()
