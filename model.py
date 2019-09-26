"""
SRNet - Editing Text in the Wild
The main SRNet model implementation.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License (see LICENSE for details)
Written by Yu Qian
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from loss import build_discriminator_loss, build_generator_loss
import cfg

class SRNet():

    def __init__(self, shape = [224, 224], name = ''):

        self.name = name
        self.cnum = 32
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.i_t = tf.placeholder(dtype = tf.float32, shape = [None] + shape + [3])
            self.i_s = tf.placeholder(dtype = tf.float32, shape = [None] + shape + [3])
            self.t_sk = tf.placeholder(dtype = tf.float32, shape = [None] + shape + [1])
            self.t_t = tf.placeholder(dtype = tf.float32, shape = [None] + shape + [3])
            self.t_b = tf.placeholder(dtype = tf.float32, shape = [None] + shape + [3])
            self.t_f = tf.placeholder(dtype = tf.float32, shape = [None] + shape + [3])
            self.mask_t = tf.placeholder(dtype = tf.float32, shape = [None] + shape + [1])
            self.global_step = tf.Variable(tf.constant(0))
            self.build_whole_net_with_loss()
            self.build_optimizer()
            self.build_summary_op()

    def _res_block(self, x, activation = tf.nn.leaky_relu, padding = 'SAME', name = 'res_block'):
        
        cnum = x.get_shape().as_list()[-1]
        xin = x
        x = tf.layers.conv2d(x, cnum // 4, kernel_size = 1, strides = 1, activation = activation, padding = padding, name = name + '_conv1')
        x = tf.layers.conv2d(x, cnum // 4, kernel_size = 3, strides = 1, activation = activation, padding = padding, name = name + '_conv2')
        x = tf.layers.conv2d(x, cnum, kernel_size = 1, strides = 1, activation = None, padding = padding, name = name + '_conv3')
        x = tf.add(xin, x, name = name + '_add')
        x = tf.layers.batch_normalization(x, name = name + '_bn')
        x = activation(x, name = name + '_out')
        return x

    def _conv_bn_relu(self, x, cnum = None, activation = tf.nn.leaky_relu, padding = 'SAME', name = 'conv_bn_relu'):
        
        cnum = x.get_shape().as_list()[-1] if cnum is None else cnum
        x = tf.layers.conv2d(x, cnum, kernel_size = 3, strides = 1, activation = None, padding = padding, name = name + '_conv')
        x = tf.layers.batch_normalization(x, name = name + '_bn')
        x = activation(x, name = name + '_out')
        return x

    def build_res_net(self, x, activation = tf.nn.leaky_relu, padding = 'SAME', name = 'res_net'):
        
        x = self._res_block(x, activation = activation, padding = padding, name = name + '_block1')
        x = self._res_block(x, activation = activation, padding = padding, name = name + '_block2')
        x = self._res_block(x, activation = activation, padding = padding, name = name + '_block3')
        x = self._res_block(x, activation = activation, padding = padding, name = name + '_block4')
        return x

    def build_encoder_net(self, x, activation = tf.nn.leaky_relu, padding = 'SAME', name = 'encoder_net', get_feature_map = False):
        
        x = self._conv_bn_relu(x, self.cnum, name = name + '_conv1_1')
        x = self._conv_bn_relu(x, self.cnum, name = name + '_conv1_2')
        
        x = tf.layers.conv2d(x, 2 * self.cnum, kernel_size = 3, strides = 2, activation = activation, padding = padding, name = name + '_pool1')
        x = self._conv_bn_relu(x, 2 * self.cnum, name = name + '_conv2_1')
        x = self._conv_bn_relu(x, 2 * self.cnum, name = name + '_conv2_2')
        f1 = x
        
        x = tf.layers.conv2d(x, 4 * self.cnum, kernel_size = 3, strides = 2, activation = activation, padding = padding, name = name + '_pool2')
        x = self._conv_bn_relu(x, 4 * self.cnum, name = name + '_conv3_1')
        x = self._conv_bn_relu(x, 4 * self.cnum, name = name + '_conv3_2')
        f2 = x
        
        x = tf.layers.conv2d(x, 8 * self.cnum, kernel_size = 3, strides = 2, activation = activation, padding = padding, name = name + '_pool3')
        x = self._conv_bn_relu(x, 8 * self.cnum, name = name + '_conv4_1')
        x = self._conv_bn_relu(x, 8 * self.cnum, name = name + '_conv4_2')
        if get_feature_map:
            return x, [f2, f1]
        else:
            return x

    def build_decoder_net(self, x, fuse = None, activation = tf.nn.leaky_relu, padding = 'SAME', name = 'decoder_net', get_feature_map = False):

        if fuse and fuse[0] is not None:
            x = tf.concat([x, fuse[0]], axis = -1, name = name + '_fuse1')
        x = self._conv_bn_relu(x, 8 * self.cnum, name = name + '_conv1_1')
        x = self._conv_bn_relu(x, 8 * self.cnum, name = name + '_conv1_2')
        f1 = x
        
        x = tf.layers.conv2d_transpose(x, 4 * self.cnum, kernel_size = 3, strides = 2, activation = activation, padding = padding, name = name + '_deconv1')
        if fuse and fuse[1] is not None:
            x = tf.concat([x, fuse[1]], axis = -1, name = name + '_fuse2')
        x = self._conv_bn_relu(x, 4 * self.cnum, name = name + '_conv2_1')
        x = self._conv_bn_relu(x, 4 * self.cnum, name = name + '_conv2_2')
        f2 = x
        
        x = tf.layers.conv2d_transpose(x, 2 * self.cnum, kernel_size = 3, strides = 2, activation = activation, padding = padding, name = name + '_deconv2')
        if fuse and fuse[2] is not None:
            x = tf.concat([x, fuse[2]], axis = -1, name = name + '_fuse3')
        x = self._conv_bn_relu(x, 2 * self.cnum, name = name + '_conv3_1')
        x = self._conv_bn_relu(x, 2 * self.cnum, name = name + '_conv3_2')
        f3 = x
        
        x = tf.layers.conv2d_transpose(x, self.cnum, kernel_size = 3, strides = 2, activation = activation, padding = padding, name = name + '_deconv3')
        x = self._conv_bn_relu(x, self.cnum, name = name + '_conv4_1')
        x = self._conv_bn_relu(x, self.cnum, name = name + '_conv4_2')
        if get_feature_map:
            return x, [f1, f2, f3]
        else:
            return x
        
    def build_text_conversion_net(self, x_t, x_s, padding = 'SAME', name = 'tcn'):
        
        x_t = self.build_encoder_net(x_t, name = name + '_t_encoder')
        x_t = self.build_res_net(x_t, name = name + '_t_res')

        x_s = self.build_encoder_net(x_s, name = name + '_s_encoder')
        x_s = self.build_res_net(x_s, name = name + '_s_res')

        x = tf.concat([x_t, x_s], axis = -1, name = name + '_concat1')
        
        y_sk = self.build_decoder_net(x, name = name + '_sk_decoder')
        y_sk_out = tf.layers.conv2d(y_sk, 1, kernel_size = 3, strides = 1, activation = 'sigmoid', padding = padding, name = name + '_sk_out')

        y_t = self.build_decoder_net(x, name = name + '_t_decoder')
        y_t = tf.concat([y_sk, y_t], axis = -1, name = name + '_concat2')
        y_t = self._conv_bn_relu(y_t, name = name + '_t_cbr')
        y_t_out = tf.layers.conv2d(y_t, 3, kernel_size = 3, strides = 1, activation = 'tanh', padding = padding, name = name + '_t_out')
        return y_sk_out, y_t_out
        
    def build_background_inpainting_net(self, x, padding = 'SAME', name = 'bin'):
        
        x, f_encoder = self.build_encoder_net(x, name = name + '_encoder', get_feature_map = True)
        x = self.build_res_net(x, name = name + '_res')
        x, fuse = self.build_decoder_net(x, fuse = [None] + f_encoder, name = name + '_decoder', get_feature_map = True)
        x = tf.layers.conv2d(x, 3, kernel_size = 3, strides = 1, activation = 'tanh', padding = padding, name = name + '_out')
        return x, fuse

    def build_fusion_net(self, x, fuse, padding = 'SAME', name = 'fn'):

        x = self.build_encoder_net(x, name = name + '_encoder')
        x = self.build_res_net(x, name = name + '_res')
        x = self.build_decoder_net(x, fuse, name = name + '_decoder')
        x = tf.layers.conv2d(x, 3, kernel_size = 3, strides = 1, activation = 'tanh', padding = padding, name = name + '_out')
        return x

    def build_discriminator(self, x, activation = tf.nn.leaky_relu, padding = 'SAME', name = 'discriminator'):
        
        with tf.variable_scope('D'):
            x = tf.layers.conv2d(x, 64, kernel_size = 3, strides = 2, activation = activation, padding = padding, name = name + '_conv1')
            x = tf.layers.conv2d(x, 128, kernel_size = 3, strides = 2, activation = None, padding = padding, name = name + '_conv2')
            x = tf.layers.batch_normalization(x, name = name + '_conv2_bn')
            x = activation(x, name = name + '_conv2_activation')
            x = tf.layers.conv2d(x, 256, kernel_size = 3, strides = 2, activation = None, padding = padding, name = name + '_conv3')
            x = tf.layers.batch_normalization(x, name = name + '_conv3_bn')
            x = activation(x, name = name + '_conv3_activation')
            x = tf.layers.conv2d(x, 512, kernel_size = 3, strides = 2, activation = None, padding = padding, name = name + '_conv4')
            x = tf.layers.batch_normalization(x, name = name + '_conv4_bn')
            x = activation(x, name = name + '_conv4_activation')
            x = tf.layers.conv2d(x, 1, kernel_size = 3, strides = 1, activation = None, padding = padding, name = name + '_conv5')
            x = tf.layers.batch_normalization(x, name = name + '_conv5_bn')
            x = tf.nn.sigmoid(x, name = '_out')
        return x

    def build_generator(self, inputs, name = 'generator'):
        
        i_t, i_s = inputs
        with tf.variable_scope('G'):
            o_sk, o_t = self.build_text_conversion_net(i_t, i_s, name = name + '_tcn')
            o_b, fuse = self.build_background_inpainting_net(i_s, name = name + '_bin')
            o_f = self.build_fusion_net(o_t, fuse, name = name + '_fn')
        return o_sk, o_t, o_b, o_f
    
    def build_whole_net_with_loss(self):

        i_t, i_s = self.i_t, self.i_s
        t_sk, t_t, t_b, t_f, mask_t = self.t_sk, self.t_t, self.t_b, self.t_f, self.mask_t
        inputs = [i_t, i_s]
        labels = [t_sk, t_t, t_b, t_f]
        
        o_sk, o_t, o_b, o_f = self.build_generator(inputs)
        self.o_sk = tf.identity(o_sk, name = 'o_sk')
        self.o_t = tf.identity(o_t, name = 'o_t')
        self.o_b = tf.identity(o_b, name = 'o_b')
        self.o_f = tf.identity(o_f, name = 'o_f')

        i_db_true = tf.concat([t_b, i_s], axis = -1, name = 'db_true_concat')
        i_db_pred = tf.concat([o_b, i_s], axis = -1, name = 'db_pred_concat')
        i_db = tf.concat([i_db_true, i_db_pred], axis = 0, name = 'db_concat')

        i_df_true = tf.concat([t_f, i_t], axis = -1, name = 'df_true_concat')
        i_df_pred = tf.concat([o_f, i_t], axis = -1, name = 'df_pred_concat')
        i_df = tf.concat([i_df_true, i_df_pred], axis = 0, name = 'df_concat')

        o_db = self.build_discriminator(i_db, name = 'db')
        o_df = self.build_discriminator(i_df, name = 'df')
            
        i_vgg = tf.concat([t_f, o_f], axis = 0, name = 'vgg_concat')

        vgg_graph_def = tf.GraphDef()
        vgg_graph_path = cfg.vgg19_weights
        with open(vgg_graph_path, 'rb') as f:
            vgg_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(vgg_graph_def, input_map = {"inputs:0": i_vgg})
        with tf.Session() as sess:
            o_vgg_1 = sess.graph.get_tensor_by_name("import/block1_conv1/Relu:0")
            o_vgg_2 = sess.graph.get_tensor_by_name("import/block2_conv1/Relu:0")
            o_vgg_3 = sess.graph.get_tensor_by_name("import/block3_conv1/Relu:0")
            o_vgg_4 = sess.graph.get_tensor_by_name("import/block4_conv1/Relu:0")
            o_vgg_5 = sess.graph.get_tensor_by_name("import/block5_conv1/Relu:0")
        
        out_g = [o_sk, o_t, o_b, o_f, mask_t]
        out_d = [o_db, o_df]
        out_vgg = [o_vgg_1, o_vgg_2, o_vgg_3, o_vgg_4, o_vgg_5]

        db_loss = build_discriminator_loss(o_db, name = 'db_loss')
        df_loss = build_discriminator_loss(o_df, name = 'df_loss')
        self.d_loss_detail = [db_loss, df_loss]
        self.d_loss = tf.add(db_loss, df_loss, name = 'd_loss')
        self.g_loss, self.g_loss_detail = build_generator_loss(out_g, out_d, out_vgg, labels, name = 'g_loss')
     
    def build_optimizer(self):

        self.learning_rate = tf.train.exponential_decay(learning_rate = cfg.learning_rate, global_step = self.global_step,
                                decay_steps = cfg.decay_steps, decay_rate = cfg.decay_rate, staircase = cfg.staircase)
        d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='D')
        with tf.control_dependencies(d_update_ops):
            self.d_train_step = tf.train.AdamOptimizer(self.learning_rate, cfg.beta1, cfg.beta2).minimize(self.d_loss,
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'D'))
        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='G')
        with tf.control_dependencies(g_update_ops):
            self.g_train_step = tf.train.AdamOptimizer(self.learning_rate, cfg.beta1, cfg.beta2).minimize(self.g_loss,
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'G'))

    def build_summary_op(self):
        
        d_summary_loss = tf.summary.scalar("loss", self.d_loss)
        d_summary_loss_db = tf.summary.scalar("l_db", self.d_loss_detail[0])
        d_summary_loss_df = tf.summary.scalar("l_df", self.d_loss_detail[1])
        
        g_summary_loss = tf.summary.scalar("loss", self.g_loss)
        g_summary_loss_t_sk = tf.summary.scalar("l_t_sk", self.g_loss_detail[0])
        g_summary_loss_t_l1 = tf.summary.scalar("l_t_l1", self.g_loss_detail[1])
        g_summary_loss_b_gan = tf.summary.scalar("l_b_gan", self.g_loss_detail[2])
        g_summary_loss_b_l1 = tf.summary.scalar("l_b_l1", self.g_loss_detail[3])
        g_summary_loss_f_gan = tf.summary.scalar("l_f_gan", self.g_loss_detail[4])
        g_summary_loss_f_l1 = tf.summary.scalar("l_f_l1", self.g_loss_detail[5])
        g_summary_loss_f_vgg_per = tf.summary.scalar("l_f_vgg_per", self.g_loss_detail[6])
        g_summary_loss_f_vgg_style = tf.summary.scalar("l_f_vgg_style", self.g_loss_detail[7])
        
        self.d_summary_op = tf.summary.merge([d_summary_loss, d_summary_loss_db, d_summary_loss_df])
        self.g_summary_op = tf.summary.merge([g_summary_loss, g_summary_loss_t_sk, g_summary_loss_t_l1,
                                              g_summary_loss_b_gan, g_summary_loss_b_l1, g_summary_loss_f_gan,
                                              g_summary_loss_f_l1, g_summary_loss_f_vgg_per, g_summary_loss_f_vgg_style])
        
        self.d_writer = tf.summary.FileWriter(os.path.join(cfg.tensorboard_dir, self.name, 'descriminator'), self.graph)
        self.g_writer = tf.summary.FileWriter(os.path.join(cfg.tensorboard_dir, self.name, 'generator'), self.graph)
     
    def train_step(self, sess, global_step, i_t, i_s, t_sk, t_t, t_b, t_f, mask_t):

        feed_dict = {
            self.i_t: i_t,
            self.i_s: i_s,
            self.t_sk: t_sk,
            self.t_t: t_t,
            self.t_b: t_b,
            self.t_f: t_f,
            self.mask_t: mask_t,
            self.global_step: global_step
        }
        
        with self.graph.as_default():
            _, d_loss, d_log = sess.run([self.d_train_step, self.d_loss, self.d_summary_op], feed_dict = feed_dict)
            _, g_loss, g_log = sess.run([self.g_train_step, self.g_loss, self.g_summary_op], feed_dict = feed_dict)
        return d_loss, g_loss, d_log, g_log
    
    def predict(self, sess, i_t, i_s, to_shape = None):
        
        assert i_t.shape == i_s.shape and i_t.dtype == i_s.dtype
        assert len(i_t.shape) == 3 or (len(i_t.shape) == 4 and to_shape is not None \
                                        and i_t.shape[1] == cfg.data_shape[0] \
                                        and i_t.shape[2] % 8 == 0 \
                                        and i_t.dtype == np.float32)
        assert i_t.dtype == np.uint8 \
               or (i_t.dtype == np.float32 and np.min(i_t) >= -1 and np.max(i_t) <= 1)

        # process raw image, len(i_t.shape) == 3
        if len(i_t.shape) == 3:
            if not to_shape:
                h, w = i_t.shape[:2]
                to_shape = (w, h) # w first for cv2
            if i_t.shape[0] != cfg.data_shape[0]:
                ratio = cfg.data_shape[0] / h
                predict_h = cfg.data_shape[0]
                predict_w = round(int(w * ratio) / 8) * 8
                predict_scale = (predict_w, predict_h) # w first for cv2
                i_t = cv2.resize(i_t, predict_scale)
                i_s = cv2.resize(i_s, predict_scale)
            if i_t.dtype == np.uint8:
                i_t = i_t.astype(np.float32) / 127.5 - 1.
                i_s = i_s.astype(np.float32) / 127.5 - 1.
            i_t = np.expand_dims(i_t, axis = 0)
            i_s = np.expand_dims(i_s, axis = 0)

        result = sess.run([self.o_sk, self.o_t, self.o_b, self.o_f], feed_dict = {self.i_t: i_t, self.i_s: i_s})
        o_sk, o_t, o_b, o_f = result
        o_sk = cv2.resize((o_sk[0] * 255.).astype(np.uint8), to_shape, interpolation=cv2.INTER_NEAREST)
        o_t = cv2.resize(((o_t[0] + 1.) * 127.5).astype(np.uint8), to_shape)
        o_b = cv2.resize(((o_b[0] + 1.) * 127.5).astype(np.uint8), to_shape)
        o_f = cv2.resize(((o_f[0] + 1.) * 127.5).astype(np.uint8), to_shape)
        return [o_sk, o_t, o_b, o_f]
