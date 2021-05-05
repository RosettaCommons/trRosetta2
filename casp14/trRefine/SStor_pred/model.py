#!/usr/bin/env python

import os
import time
import json
import tensorflow as tf
import numpy as np
from utils import *
from data_loader import *
from attention import attention
from ffindex import FFindexDB, read_index, read_data

eps  = 1e-9 # small number

N_AA = 20 # regular aa
N_AA_MSA = 21 # regular aa + gap
WMIN = 0.8

N_PRINT_LEVEL = 50

TRAIN_LOG = "Train [%03d/%03d] counter: %5d  time: %10.1f lr: %.6f | loss: %7.4f | %.4f %.4f %.4f %.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f %.4f %.4f %.4f"
VALID_LOG = "Valid [%03d/%03d] counter: %5d  time: %10.1f lr: %.6f | loss: %7.4f | %.4f %.4f %.4f %.4f %.4f %.4f %.4f | %.4f %.4f %.4f %.4f %.4f %.4f %.4f"

# ResNet model definition
class ResNet_model(object):
    def __init__(self, sess, n_1d_layer=2, dilation=[1],
                 p_dropout=0.2, l2_coeff=0.001, kernel_size=3, n_hidden_rnn=64, attention_size=50,
                 n_feat_1d=64, n_feat_2d = 64, n_bottle_1d=32, n_bottle_2d=32, use_cpu=False, use_templ=False):
        self.sess = sess # tensorflow session
        #
        self.n_1d_layer = n_1d_layer
        self.dilation = dilation
        self.kernel = kernel_size
        #
        self.n_seq_1d = N_AA + N_AA_MSA + 1
        self.n_str_feat = 8 # 1 dist_map + 6 ori_map + 1 seqsep
        self.SS_dim     = 9
        self.phi_dim    = 36
        self.psi_dim    = 36
        self.omg_dim    = 2
        #
        self.n_hidden_rnn = n_hidden_rnn
        self.attention_size = attention_size
        #
        # hidden layer (1d)
        self.n_feat_1d = n_feat_1d
        self.n_feat_2d = n_feat_2d
        self.n_bottle_1d = n_bottle_1d
        self.n_bottle_2d = n_bottle_2d
        #
        self.p_dropout = p_dropout
        self.rnn_p_dropout = p_dropout
        self.l2_coeff = l2_coeff
        #
        with tf.variable_scope("SStorPred"):
            self.build_model(use_cpu=use_cpu, use_templ=use_templ)

    def build_model(self, use_cpu=False, use_templ=False):
        #
        # Receive inputs
        with tf.variable_scope("input"):
            self.seq      = tf.placeholder(tf.float32, [None, N_AA], name="seq") # n_res, N_AA (blosum)
            self.msa      = tf.placeholder(tf.uint8, [None, None], name="msa") # n_seq, n_res
            self.str_2d   = tf.placeholder(tf.float32, [None, None, None, self.n_str_feat], name="str_2d") # 1, n_res, n_res, n_feat
            self.pth_1d   = tf.placeholder(tf.float32, [None, None, 4])
            self.pth_2d   = tf.placeholder(tf.float32, [None, None, None, 6])
            self.SS       = tf.placeholder(tf.float32, [None, None, self.SS_dim]) # SS label
            self.phi      = tf.placeholder(tf.float32, [None, None, self.phi_dim]) # ref. phi distrib
            self.psi      = tf.placeholder(tf.float32, [None, None, self.psi_dim]) # ref. psi distrib
            self.omg      = tf.placeholder(tf.float32, [None, None, self.omg_dim]) # ref. omg distrib
            self.is_train = tf.placeholder(tf.bool)
            self.n_batch = tf.shape(self.str_2d)[0]
            self.n_res = tf.shape(self.seq)[0]
        #
        # 3-state answer
        self.SS3 = tf.stack([tf.reduce_sum(self.SS[:,:, :3], axis=-1), 
                             tf.reduce_sum(self.SS[:,:,3:6], axis=-1),
                             tf.reduce_sum(self.SS[:,:,6: ], axis=-1)], axis=-1)
        #
        # 8-state answer
        self.SS8 = tf.stack([self.SS[:,:,0],
                             tf.reduce_sum(self.SS[:,:,1:3], axis=-1),
                             self.SS[:,:,3],
                             self.SS[:,:,4],
                             self.SS[:,:,5],
                             self.SS[:,:,6],
                             self.SS[:,:,7],
                             self.SS[:,:,8]], axis=-1)
        #================================
        # sequence features
        #================================
        # get pssm features from MSA
        msa1hot = tf.one_hot(self.msa, N_AA_MSA, dtype=tf.float32)
        w_seq = reweight_seq(msa1hot, WMIN)
        pssm = msa2pssm(msa1hot, w_seq)
        #
        seq_1d = tf.concat([self.seq, pssm], axis=-1) # sequence based features
        seq_1d = tf.expand_dims(seq_1d, 0)
        #
        # projection to n_feat_1d
        feat = tf.layers.conv1d(seq_1d, self.n_feat_1d, 1, padding='same')
        #
        #=================================
        # 1D ResNet with combined features
        #=================================
        # Stacking 1-dim residual blocks
        for i in range(self.n_1d_layer):
            d = self.dilation[i%len(self.dilation)]
            feat = self.ResNet_block_1d(feat, self.is_train, step=i, dilation=d)
        feat = tf.nn.elu(inst_norm(feat))
        #
        seq_1d = tf.tile(tf.reshape(feat, [-1]), [self.n_batch*self.n_res])
        seq_1d = tf.reshape(seq_1d, [self.n_batch, self.n_res, self.n_res, self.n_feat_1d])
        
        if use_templ:
            #=================================
            # Process template info
            #=================================
            t1d = tf.concat([tf.sin(self.pth_1d[:,:,0])[:,:,None],
                            tf.cos(self.pth_1d[:,:,0])[:,:,None],
                            tf.sin(self.pth_1d[:,:,1])[:,:,None],
                            tf.cos(self.pth_1d[:,:,1])[:,:,None],
                            self.pth_1d[:,:,2:]], axis=-1)
            t2d = tf.concat([tf.one_hot(tf.cast(self.pth_2d[:,:,:,0],dtype=tf.uint8),19,dtype=tf.float32),
                            tf.sin(self.pth_2d[:,:,:,1])[:,:,:,None],
                            tf.cos(self.pth_2d[:,:,:,1])[:,:,:,None],
                            tf.sin(self.pth_2d[:,:,:,2])[:,:,:,None],
                            tf.cos(self.pth_2d[:,:,:,2])[:,:,:,None],
                            tf.sin(self.pth_2d[:,:,:,3])[:,:,:,None],
                            tf.cos(self.pth_2d[:,:,:,3])[:,:,:,None],
                            tf.tile(t1d[:,:,None,:],[1,1,self.n_res,1]),
                            tf.tile(t1d[:,None,:,:],[1,self.n_res,1,1])], axis=-1)
            sgnl = tf.nn.elu(tf.layers.conv2d(t2d, self.n_feat_2d, 3, padding='SAME'))
            prob = tf.nn.softmax(tf.layers.conv2d(t2d, 1, 3, padding='SAME'), axis=0)
            t2d = tf.reduce_sum(sgnl*prob,axis=0)
        #
        # combine with 2D str features
        str_2d = tf.layers.conv2d(self.str_2d, self.n_feat_2d, 1, padding='same')
        str_2d = tf.nn.elu(inst_norm(str_2d))
        
        if use_templ:
            feat = tf.concat((str_2d, t2d[None, :,:,:], seq_1d, tf.transpose(seq_1d, (0,2,1,3))), axis=-1)
        else:
            feat = tf.concat((str_2d, seq_1d, tf.transpose(seq_1d, (0,2,1,3))), axis=-1)
        #
        # projection to n_feat_2d
        feat = tf.layers.conv2d(feat, self.n_feat_2d, 1, padding='same', use_bias=False)
        #
        #=================================
        # 2D ResNet with combined features
        #=================================
        # Stacking 2-dim residual blocks (receptive field size: 61) 
        for i in range(8):
            d = self.dilation[i%len(self.dilation)]
            feat = self.ResNet_block_2d(feat, self.is_train, step=i, dilation=d)
        feat = tf.nn.elu(inst_norm(feat))
        #
        #=================================
        # LSTM to extract 1-dimensional features from input features
        #=================================
        with tf.variable_scope("AttBiLSTM") as scope:
            # convert to 1D using LSTM
            weights = {'out': tf.Variable(tf.random_normal([self.n_hidden_rnn*2, self.n_feat_1d]), name='kernel')}
            biases  = {'out': tf.Variable(tf.random_normal([self.n_feat_1d]), name='bias')}
            #
            if use_cpu:
                feat, self.alphas = self.BiLSTM_w_attention_cpu(feat, weights, biases)
            else:
                feat, self.alphas = self.BiLSTM_w_attention_gpu(feat, weights, biases)
            feat = tf.reshape(feat, [self.n_batch, self.n_res, self.n_feat_1d])
        #
        # Stacking 1-dim residual blocks
        with tf.variable_scope("additional_ResNet"):
            for i in range(4):
                d = self.dilation[i%len(self.dilation)]
                feat = self.ResNet_block_1d(feat, self.is_train, step=i, dilation=d)
        #
        # Final branching
        with tf.variable_scope("final_SS") as scope:
            for i in range(4):
                d = self.dilation[i%len(self.dilation)]
                feat = self.ResNet_block_1d(feat, self.is_train, step=i, dilation=d)
            feat = tf.nn.elu(feat)
            SS_logit = tf.layers.conv1d(feat, self.SS_dim, 1, padding='same')
        #
        with tf.variable_scope("final_phi") as scope:
            for i in range(4):
                d = self.dilation[i%len(self.dilation)]
                feat = self.ResNet_block_1d(feat, self.is_train, step=i, dilation=d)
            feat = tf.nn.elu(feat)
            phi_logit = tf.layers.conv1d(feat, self.phi_dim, 1, padding='same')
        #
        with tf.variable_scope("final_psi") as scope:
            for i in range(4):
                d = self.dilation[i%len(self.dilation)]
                feat = self.ResNet_block_1d(feat, self.is_train, step=i, dilation=d)
            feat = tf.nn.elu(feat)
            psi_logit = tf.layers.conv1d(feat, self.psi_dim, 1, padding='same')
        #
        with tf.variable_scope("final_omg") as scope:
            for i in range(4):
                d = self.dilation[i%len(self.dilation)]
                feat = self.ResNet_block_1d(feat, self.is_train, step=i, dilation=d)
            feat = tf.nn.elu(feat)
            omg_logit = tf.layers.conv1d(feat, self.omg_dim, 1, padding='same')
        #
        # calculate probability
        SS_prob = tf.nn.softmax(SS_logit)
        SS3_prob = tf.stack([tf.reduce_sum(SS_prob[:,:, :3], axis=-1), 
                             tf.reduce_sum(SS_prob[:,:,3:6], axis=-1),
                             tf.reduce_sum(SS_prob[:,:,6: ], axis=-1)], axis=-1)
        SS8_prob = tf.stack([SS_prob[:,:,0],
                             tf.reduce_sum(SS_prob[:,:,1:3], axis=-1),
                             SS_prob[:,:,3],
                             SS_prob[:,:,4],
                             SS_prob[:,:,5],
                             SS_prob[:,:,6],
                             SS_prob[:,:,7],
                             SS_prob[:,:,8]], axis=-1)
        phi_prob = tf.nn.softmax(phi_logit)
        psi_prob = tf.nn.softmax(psi_logit)
        omg_prob = tf.nn.softmax(omg_logit)
        self.prob_s = [SS_prob, SS8_prob, SS3_prob, phi_prob, psi_prob, omg_prob]
        
        # calculate loss function (softmax cross-entropy)
        # For SS & omega, it is same as categorical cross entropy
        # For phi psi angles, reference is defined with von Mises distrib.
        # It should be noted that minimizing softmax cross-entropy is same as minimizing KL divergence
        
        SS_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(self.SS, SS_logit))

        SS3_loss = -tf.reduce_sum(self.SS3 * tf.log(SS3_prob+eps), axis=-1)
        SS3_loss = tf.reduce_mean(SS3_loss)
        
        SS8_loss = -tf.reduce_sum(self.SS8 * tf.log(SS8_prob+eps), axis=-1)
        SS8_loss = tf.reduce_mean(SS8_loss)
        
        phi_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(self.phi, phi_logit))
        psi_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(self.psi, psi_logit))
        omg_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(self.omg, omg_logit))

        self.t_vars = tf.trainable_variables()
        #
        # L2-regularization to avoid overfitting
        l2_loss = tf.add_n([tf.nn.l2_loss(var)
                            for var in self.t_vars if 'kernel' in var.name]) * self.l2_coeff
        #
        # total losses
        self.loss_s = [SS_loss, SS8_loss, SS3_loss, phi_loss, psi_loss, omg_loss, l2_loss]
        self.tot_loss = tf.add_n(self.loss_s)
        
        # calculate accuracy
        equal = tf.cast(tf.equal(tf.argmax(SS_prob, axis=-1), tf.argmax(self.SS, axis=-1)), tf.float32)
        SS_acc = tf.reduce_mean(tf.cast(equal, tf.float32))
        #
        equal = tf.cast(tf.equal(tf.argmax(SS3_prob, axis=-1), tf.argmax(self.SS3, axis=-1)), tf.float32)
        SS3_acc = tf.reduce_mean(tf.cast(equal, tf.float32))
        #
        equal = tf.cast(tf.equal(tf.argmax(SS8_prob, axis=-1), tf.argmax(self.SS8, axis=-1)), tf.float32)
        SS8_acc = tf.reduce_mean(tf.cast(equal, tf.float32))
        #
        phi_equal = tf.equal(tf.argmax(phi_prob, axis=-1), tf.argmax(self.phi, axis=-1))
        phi_acc = tf.reduce_mean(tf.cast(phi_equal, tf.float32))
        #
        psi_equal = tf.equal(tf.argmax(psi_prob, axis=-1), tf.argmax(self.psi, axis=-1))
        psi_acc = tf.reduce_mean(tf.cast(psi_equal, tf.float32))
        #
        omg_equal = tf.equal(tf.argmax(omg_prob, axis=-1), tf.argmax(self.omg, axis=-1))
        omg_acc = tf.reduce_mean(tf.cast(omg_equal, tf.float32))
        #
        equal = tf.stack([phi_equal, psi_equal, omg_equal], axis=-1)
        equal = tf.reduce_all(equal, axis=-1)
        tot_acc = tf.reduce_mean(tf.cast(equal, tf.float32))
        self.acc_s = [SS_acc, SS8_acc, SS3_acc, phi_acc, psi_acc, omg_acc, tot_acc]
        #
        # define trained model saver
        self.saver = tf.train.Saver()
    
    def BiLSTM_w_attention_gpu(self, x, weights, biases, return_alphas=True):
        # prepare input data shape to match rnn function requirements
        # input x: (batch_size, n_res, n_res, n_ch)
        # Required shape: timesteps tensors list of shape (batch_size', n_input')
        #   - howto? batch_size' = batch_size*n_res , n_input' = n_ch
        #            # of timesteps = n_res
        
        # reshape input x to (batch_size*n_res, n_res, n_ch), CAUTION: batch_size=1 here
        # first n_res: time-series & time-series should go first in cudnn version
        x = tf.transpose(x, (1,0,2,3))
        x = tf.reshape(x, [self.n_res, self.n_batch*self.n_res, self.n_feat_2d])
        #x = tf.reshape(x, [self.n_res, self.n_batch*self.n_res, self.n_hidden_rnn])
        
        # define a lstm cell
        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(3, self.n_hidden_rnn, direction='bidirectional', dtype=tf.float32)

        # Get BiLSTM cell output 
        outputs, states = lstm(x)
        outputs = tf.concat(outputs, axis=-1)

        # apply attention
        outputs, alphas = attention(outputs, self.attention_size, return_alphas=return_alphas, time_major=True)

        # apply linear activation
        return tf.matmul(outputs, weights['out']) + biases['out'], alphas
    
    def BiLSTM_w_attention_cpu(self, x, weights, biases, return_alphas=True):
        # prepare input data shape to match rnn function requirements
        # input x: (batch_size, n_res, n_res, n_ch)
        # Required shape: timesteps tensors list of shape (batch_size', n_input')
        #   - howto? batch_size' = batch_size*n_res , n_input' = n_ch
        #            # of timesteps = n_res
        
        # reshape input x to (batch_size*n_res, n_res, n_ch), CAUTION: batch_size=1 here
        # first n_res: time-series & time-series should go first in cudnn version
        x = tf.transpose(x, (1,0,2,3))
        x = tf.reshape(x, [self.n_res, self.n_batch*self.n_res, self.n_feat_2d])
        #x = tf.reshape(x, [self.n_res, self.n_batch*self.n_res, self.n_hidden_rnn])
        
        # define a lstm cell
        with tf.variable_scope("cudnn_lstm"):
            single_cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.n_hidden_rnn)
            cells_fw = [single_cell() for _ in range(3)]
            cells_bw = [single_cell() for _ in range(3)]
            #
            # Get BiLSTM cell output 
            outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw, cells_bw, x, time_major=True, dtype=tf.float32)
        outputs = tf.concat(outputs, axis=-1)

        # apply attention
        outputs, alphas = attention(outputs, self.attention_size, return_alphas=return_alphas, time_major=True)

        # apply linear activation
        return tf.matmul(outputs, weights['out']) + biases['out'], alphas
    
    def ResNet_block_2d(self, x, is_train, step=0, dilation=1): # bottleneck block w/ pre-activation
        with tf.variable_scope("ResNet_2d_{}".format(step)) as scope:
            shortcut = x
            # bottleneck layer (kernel: 1, n_feat_2d => n_bottle_2d)
            x = tf.nn.elu(inst_norm(x))
            x = tf.layers.conv2d(x, self.n_bottle_2d, 1, padding='same')
            x = tf.nn.elu(inst_norm(x))
            # convolution
            x = tf.layers.conv2d(x, self.n_bottle_2d, self.kernel, dilation_rate=dilation,
                                 padding='same')
            x = tf.nn.elu(inst_norm(x))
            x = tf.layers.dropout(x, rate=self.p_dropout, training=is_train)
            # project up (kernel: 1, n_bottle_1d => n_feat_1d)
            x = tf.layers.conv2d(x, self.n_feat_2d, 1, padding='same')
            # add
            x += shortcut
        return x
    
    def ResNet_block_1d(self, x, is_train, step=0, dilation=1): # bottleneck block w/ pre-activation
        with tf.variable_scope("ResNet_1d_{}".format(step)) as scope:
        #with tf.variable_scope("ResNet_1d_{}".format(step), custom_getter=float32_variable_storage_getter) as scope:
            shortcut = x
            # bottleneck layer (kernel: 1, n_feat_1d => n_bottle_1d)
            x = tf.nn.elu(inst_norm(x))
            x = tf.layers.conv1d(x, self.n_bottle_1d, 1, padding='same')
            x = tf.nn.elu(inst_norm(x))
            # convolution
            x = tf.layers.conv1d(x, self.n_bottle_1d, self.kernel, dilation_rate=dilation,
                                 padding='same')
            x = tf.nn.elu(inst_norm(x))
            x = tf.layers.dropout(x, rate=self.p_dropout, training=is_train)
            # project up (kernel: 1, n_bottle_1d => n_feat_1d)
            x = tf.layers.conv1d(x, self.n_feat_1d, 1, padding='same')
            # add
            x += shortcut
        return x
    
    def save(self, folder, prefix):
        if not os.path.exists(folder):
            os.mkdir(folder)
        self.saver.save(self.sess, folder+"/%s.ckpt"%prefix)
    
    def load(self, folder, prefix):
        model_fn = os.path.join(folder, "%s.ckpt.index"%prefix)
        if os.path.exists(model_fn):
            self.saver.restore(self.sess, folder+"/%s.ckpt"%prefix)
            return True
        return False
    
    def close(self):
        self.sess.close()

    def train(self, config_file):
        # read config_file
        with open(config_file) as json_file:
            config = json.load(json_file, object_hook=Json_param)
        #
        train_pdbs = [line.split()[0] for line in open(config.train_list)]
        self.n_train = len(train_pdbs)
        #
        global_step = tf.Variable(0, trainable=False)
        
        # define optimizer
        if config.lr_schedule == "CosineDecay":
            lr = tf.train.cosine_decay_restarts(learning_rate=config.lr,
                                                global_step=global_step,
                                                first_decay_steps=config.f_decay*self.n_train,
                                                t_mul=config.t_mul,
                                                m_mul=config.m_mul,
                                                alpha=0.1)
        elif config.lr_schedule == 'ExpDecay':
            lr = tf.train.exponential_decay(config.lr,
                                            global_step=global_step,
                                            decay_steps=self.n_train,
                                            decay_rate=0.99,
                                            staircase=True)
        else: # flat lr
            lr = tf.Variable(config.lr, trainable=False)
        
        if config.optim == 'adam':
            optimizer = tf.train.AdamOptimizer(lr)
        elif config.optim == 'momentum':
            optimizer = tf.train.MomentumOptimizer(lr, 0.5, use_nesterov=True)
        else:
            optimizer = tf.train.GradientDescentOptimizer(lr)

        optim = optimizer.minimize(self.tot_loss, global_step=global_step, var_list=self.t_vars)
        ops_to_run = [optim, lr, self.tot_loss, self.loss_s, self.acc_s]
        #
        # initialize all variables
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)
        #
        counter = 0
        self.start_time = time.time()
        #
        n_batch = len(train_pdbs) 
        min_val_loss = config.best_loss
        #
        # Try to load pre-trained model if exists
        could_load = self.load("model_%d_%d_%d"%(self.n_hidden_rnn, self.n_1d_layer, self.attention_size), 'last_epoch')
        if not could_load:
            could_load = self.load("model_%d_%d_%d"%(self.n_hidden_rnn, self.n_1d_layer, self.attention_size), 'model')
        #
        for epoch in range(config.n_epoch):
            np.random.shuffle(train_pdbs)
            tot_loss_value = 0.0
            tot_loss_s = np.zeros(7, dtype=np.float32)
            tot_acc_s = np.zeros(7, dtype=np.float32)
            n_tot = 0.0
            #
            for pdb in train_pdbs:
                seq, msa, str_2d, pth_1d, pth_2d, SS_labels, phi_labels, psi_labels, omg_labels = load_train_data(pdb, \
                                                is_train=True, mask_diag=config.mask_diag)
                #
                _, decayed_lr, loss_value, loss_s, acc_s = self.sess.run(ops_to_run,
                                         feed_dict={
                                             self.seq: seq,
                                             self.msa: msa,
                                             self.str_2d: str_2d[np.newaxis,:,:,:],
                                             self.pth_1d: pth_1d,
                                             self.pth_2d: pth_2d,
                                             self.SS: SS_labels[np.newaxis,:,:],
                                             self.phi: phi_labels[np.newaxis,:,:],
                                             self.psi: psi_labels[np.newaxis,:,:],
                                             self.omg: omg_labels[np.newaxis,:],
                                             self.is_train: True})
                tot_loss_value += loss_value
                tot_loss_s += np.array(loss_s)
                tot_acc_s += np.array(acc_s)
                n_tot += 1.0
                #
                counter += 1
                if counter % N_PRINT_LEVEL == 0:
                    loss_value = tot_loss_value/n_tot
                    loss_s = tot_loss_s/n_tot
                    acc_s = tot_acc_s/n_tot
                    tot_loss_value = 0.0
                    tot_loss_s = np.zeros(7, dtype=np.float32)
                    tot_acc_s = np.zeros(7, dtype=np.float32)
                    n_tot = 0.0
                    log_list = [epoch, config.n_epoch, counter, time.time()-self.start_time, decayed_lr*100.0, loss_value-loss_s[-1]]
                    log_list.extend(loss_s)
                    log_list.extend(acc_s)
                    print (TRAIN_LOG%tuple(log_list))
            #
            val_loss = self.validation(config, epoch, counter, decayed_lr)
            if val_loss < min_val_loss:
                self.save("model_%d_%d_%d"%(self.n_hidden_rnn, self.n_1d_layer, self.attention_size), 'model')
                min_val_loss = val_loss
            self.save("model_%d_%d_%d"%(self.n_hidden_rnn, self.n_1d_layer, self.attention_size), 'last_epoch')

    def validation(self, config, epoch, counter, decayed_lr):
        ops_to_run = [self.tot_loss, self.loss_s, self.acc_s]
        valid_pdbs = [line.split()[0] for line in open(config.valid_list) if line[0] != "#"]
        #
        tot_loss_value = 0.0
        tot_loss_s = np.zeros(7, dtype=np.float32)
        tot_acc_s = np.zeros(7, dtype=np.float32)
        n_tot = 0.0
        for pdb in valid_pdbs:
            seq, msa, str_2d, pth_1d, pth_2d, SS_labels, phi_labels, psi_labels, omg_labels\
                     = load_train_data(pdb, \
                                            is_train=False, mask_diag=config.mask_diag)
            #
            loss_value, loss_s, acc_s = self.sess.run(ops_to_run,
                                             feed_dict={
                                                 self.seq: seq,
                                                 self.msa: msa,
                                                 self.str_2d: str_2d[np.newaxis,:,:,:],
                                                 self.pth_1d: pth_1d,
                                                 self.pth_2d: pth_2d,
                                                 self.SS: SS_labels[np.newaxis,:,:],
                                                 self.phi: phi_labels[np.newaxis,:,:],
                                                 self.psi: psi_labels[np.newaxis,:,:],
                                                 self.omg: omg_labels[np.newaxis,:],
                                                 self.is_train:False})
            tot_loss_value += loss_value
            tot_loss_s += np.array(loss_s)
            tot_acc_s += np.array(acc_s)
            n_tot += 1.0
        #
        loss_value = tot_loss_value/n_tot
        loss_s = tot_loss_s/n_tot
        acc_s = tot_acc_s/n_tot
        #
        log_list = [epoch, config.n_epoch, counter, time.time()-self.start_time, decayed_lr*100.0, loss_value-loss_s[-1]]
        log_list.extend(loss_s)
        log_list.extend(acc_s)
        print (VALID_LOG%tuple(log_list))
        return loss_value-loss_s[-1]
    
    def predict(self, config):
        self.load(config.model_dir, 'model')
        #
        ffdb = None
        if config.templ_fn != None:
            ffdb = FFindexDB(read_index(config.TMPDB+'_pdb.ffindex'),
                             read_data(config.TMPDB+"_pdb.ffdata"))
            hhr_fn = config.templ_fn
        #
        out_fn = "%s.npz"%config.outprefix
        a3m_fn = config.a3m_fn
        pdb_fn = config.pdb_fn
        hhr_fn = config.templ_fn
        # 
        seq, msa, str_2d, pth_1d, pth_2d = make_input_features(a3m_fn, pdb_fn, hhr_fn, ffdb)
        #
        if hhr_fn != None:
            prob_s, alpha = self.sess.run([self.prob_s, self.alphas], feed_dict={
                                                     self.seq: seq,
                                                     self.msa: msa,
                                                     self.str_2d: str_2d[np.newaxis,:,:,:],
                                                     self.pth_1d: pth_1d,
                                                     self.pth_2d: pth_2d,
                                                     self.is_train:False})
        else:
            prob_s, alpha = self.sess.run([self.prob_s, self.alphas], feed_dict={
                                                     self.seq: seq,
                                                     self.msa: msa,
                                                     self.str_2d: str_2d[np.newaxis,:,:,:],
                                                     self.is_train:False})

        SS_prob = prob_s[0].reshape(-1, self.SS_dim)
        phi_prob = prob_s[3].reshape(-1, self.phi_dim)
        psi_prob = prob_s[4].reshape(-1, self.psi_dim)
        omg_prob = prob_s[5].reshape(-1, self.omg_dim)
        tor_prob = np.concatenate((phi_prob, psi_prob, omg_prob), axis=-1)
        np.savez_compressed(out_fn, ss9=SS_prob.astype(np.float16), tor=tor_prob.astype(np.float16))
    
    def predict_multi(self, config, a3m_s, pdb_s, hhr_s, out_s):
        self.load(config.model_dir, 'model')
        #
        ffdb = None
        if hhr_s[0] != None:
            ffdb = FFindexDB(read_index(config.TMPDB+'_pdb.ffindex'),
                             read_data(config.TMPDB+"_pdb.ffdata"))
        #
        for i, a3m_fn in enumerate(a3m_s):
            out_fn = out_s[i]
            pdb_fn = pdb_s[i]
            hhr_fn = hhr_s[i]
            if os.path.exists(out_fn):
                continue
            print ("Running..., %s"%out_fn)
            # 
            seq, msa, str_2d, pth_1d, pth_2d = make_input_features(a3m_fn, pdb_fn, hhr_fn, ffdb)
            #
            if hhr_fn != None:
                prob_s, alpha = self.sess.run([self.prob_s, self.alphas], feed_dict={
                                                         self.seq: seq,
                                                         self.msa: msa,
                                                         self.str_2d: str_2d[np.newaxis,:,:,:],
                                                         self.pth_1d: pth_1d,
                                                         self.pth_2d: pth_2d,
                                                         self.is_train:False})
            else:
                prob_s, alpha = self.sess.run([self.prob_s, self.alphas], feed_dict={
                                                         self.seq: seq,
                                                         self.msa: msa,
                                                         self.str_2d: str_2d[np.newaxis,:,:,:],
                                                         self.is_train:False})

            SS_prob = prob_s[0].reshape(-1, self.SS_dim)
            phi_prob = prob_s[3].reshape(-1, self.phi_dim)
            psi_prob = prob_s[4].reshape(-1, self.psi_dim)
            omg_prob = prob_s[5].reshape(-1, self.omg_dim)
            tor_prob = np.concatenate((phi_prob, psi_prob, omg_prob), axis=-1)
            np.savez_compressed(out_fn, ss9=SS_prob.astype(np.float16), tor=tor_prob.astype(np.float16))
