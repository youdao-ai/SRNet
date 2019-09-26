"""
SRNet - Editing Text in the Wild
Definition of loss functions.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import tensorflow as tf
import cfg

def build_discriminator_loss(x, name = 'd_loss'):

    x_true, x_pred = tf.split(x, 2, name = name + '_split')
    d_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(x_true, cfg.epsilon, 1.0)) \
                + tf.log(tf.clip_by_value(1.0 - x_pred, cfg.epsilon, 1.0)))
    return d_loss

def build_dice_loss(x_t, x_o, name = 'dice_loss'):
       
    intersection = tf.reduce_sum(x_t * x_o, axis = [1,2,3])
    union = tf.reduce_sum(x_t, axis = [1,2,3]) + tf.reduce_sum(x_o, axis = [1,2,3])
    return 1. - tf.reduce_mean((2. * intersection + cfg.epsilon)/(union + cfg.epsilon), axis = 0)

def build_l1_loss(x_t, x_o, name = 'l1_loss'):
        
    return tf.reduce_mean(tf.abs(x_t - x_o))

def build_l1_loss_with_mask(x_t, x_o, mask, name = 'l1_loss'):
    
    mask_ratio = 1. - tf.reduce_sum(mask) / tf.cast(tf.size(mask), tf.float32)
    l1 = tf.abs(x_t - x_o)
    return mask_ratio * tf.reduce_mean(l1 * mask) + (1. - mask_ratio) * tf.reduce_mean(l1 * (1. - mask))

def build_perceptual_loss(x, name = 'per_loss'):
        
    l = []
    for i, f in enumerate(x):
        l.append(build_l1_loss(f[0], f[1], name = name + '_l1_' + str(i + 1)))
    l = tf.stack(l, axis = 0, name = name + '_stack')
    l = tf.reduce_sum(l, name = name + '_sum')
    return l

def build_gram_matrix(x, name = 'gram_matrix'):
        
    x_shape = tf.shape(x)
    h, w, c = x_shape[1], x_shape[2], x_shape[3]
    matrix = tf.reshape(x, shape = [-1, h * w, c])
    gram = tf.matmul(matrix, matrix, transpose_a = True) / tf.cast(h * w * c, tf.float32)
    return gram

def build_style_loss(x, name = 'style_loss'):
        
    l = []
    for i, f in enumerate(x):
        f_shape = tf.size(f[0])
        f_norm = 1. / tf.cast(f_shape, tf.float32)
        gram_true = build_gram_matrix(f[0], name = name + '_gram_true_' + str(i + 1))
        gram_pred = build_gram_matrix(f[1], name = name + '_gram_pred_' + str(i + 1))
        l.append(f_norm * (build_l1_loss(gram_true, gram_pred, name = name + '_l1_' + str(i + 1))))
    l = tf.stack(l, axis = 0, name = name + '_stack')
    l = tf.reduce_sum(l, name = name + '_sum')
    return l

def build_vgg_loss(x, name = 'vgg_loss'):
        
    splited = []
    for i, f in enumerate(x):
        splited.append(tf.split(f, 2, name = name + '_split_' + str(i + 1)))
    l_per = build_perceptual_loss(splited, name = name + '_per')
    l_style = build_style_loss(splited, name = name + '_style')
    return l_per, l_style

def build_gan_loss(x, name = 'gan_loss'):
    
    x_true, x_pred = tf.split(x, 2, name = name + '_split')
    gan_loss = -tf.reduce_mean(tf.log(tf.clip_by_value(x_pred, cfg.epsilon, 1.0)))
    return gan_loss

def build_generator_loss(out_g, out_d, out_vgg, labels, name = 'g_loss'):
        
    o_sk, o_t, o_b, o_f, mask_t = out_g
    o_db, o_df = out_d
    o_vgg = out_vgg
    t_sk, t_t, t_b, t_f = labels

    l_t_sk = cfg.lt_alpha * build_dice_loss(t_sk, o_sk, name = name + '_dice_loss')
    l_t_l1 = build_l1_loss_with_mask(t_t, o_t, mask_t, name = name + '_lt_l1_loss')
    l_t =  l_t_l1 + l_t_sk

    l_b_gan = build_gan_loss(o_db, name = name + '_lb_gan_loss')
    l_b_l1 = cfg.lb_beta * build_l1_loss(t_b, o_b, name = name + '_lb_l1_loss')
    l_b = l_b_gan + l_b_l1
    
    l_f_gan = build_gan_loss(o_df, name = name + '_lf_gan_loss')
    l_f_l1 = cfg.lf_theta_1 * build_l1_loss(t_f, o_f, name = name + '_lf_l1_loss')
    l_f_vgg_per, l_f_vgg_style = build_vgg_loss(o_vgg, name = name + '_lf_vgg_loss')
    l_f_vgg_per = cfg.lf_theta_2 * l_f_vgg_per
    l_f_vgg_style = cfg.lf_theta_3 * l_f_vgg_style
    l_f = l_f_gan + l_f_l1 + l_f_vgg_per + l_f_vgg_style
    
    l = cfg.lt * l_t + cfg.lb * l_b + cfg.lf * l_f
    return l, [l_t_sk, l_t_l1, l_b_gan, l_b_l1, l_f_gan, l_f_l1, l_f_vgg_per, l_f_vgg_style]
