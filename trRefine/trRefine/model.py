#!/usr/bin/env python

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from data_loader import load_train_data, process_inputs, setup_valid_dataset, load_valid_data
from resnet import build_resnet
from utils import *

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(allow_growth=True)
)

N_AA_MSA = 21
WMIN = 0.8
N_DIST_BIN = 37
N_BIN360 = 37
N_BIN180 = 19

N_PRINT_LEVEL = 50

TRAIN_LOG = "Train [%03d/%03d] counter: %5d  time: %10.1f lr: %.6f | loss: %7.4f | %.4f %.4f %.4f %.4f %.4f"
VALID_LOG = "Valid [%03d/%03d] counter: %5d  time: %10.1f lr: %.6f | loss: %7.4f | %.4f %.4f %.4f %.4f %.4f"

class TrRefiner:
    def __init__(self, sess, num_chunks=5, n2d_filters=64, pdb_in=False, train=False, config_file=None, attention=True): # TODO: change mode...
        self.sess = sess
        self.num_chunks = num_chunks
        self.n2d_filters = n2d_filters
        self.half_filters = max(n2d_filters // 2, 32)
        self.p_dropout = 0.15
        self.l2_coeff = 0.00001
        #
        self.use_attention = attention
        self.crd_size = N_DIST_BIN + N_BIN180 + N_BIN360*2 + 1 # dist + orien + seqsep
        self.str_size = 1 + 6 + 15 + 1 # dist + sin/cos theta,omega,phi + estogram + seqsep
        #
        if train:
            with open(config_file) as json_file:
                config = json.load(json_file, object_hook=Json_param)
            train_pdbs = [line.split()[0] for line in open(config.train_list)]
            trainset = tf.data.Dataset.from_tensor_slices(train_pdbs)
            trainset = trainset.shuffle(len(train_pdbs), reshuffle_each_iteration=True)
            trainset = trainset.prefetch(tf.data.experimental.AUTOTUNE)
            trainset = trainset.map(lambda x: tf.compat.v1.py_func(load_train_data, [x], \
                                    [tf.uint8, tf.float32, tf.float32, tf.float32, tf.uint8, tf.uint8, tf.uint8, tf.uint8]))
                                    #msa       SS          maps        strs        labels (dist, theta, phi, omega)
            valid_pdbs = [line.split()[0] for line in open(config.valid_list)]
            setup_valid_dataset(config.valid_list)
            validset = tf.data.Dataset.from_tensor_slices(valid_pdbs)
            validset = validset.prefetch(tf.data.experimental.AUTOTUNE)
            validset = validset.map(lambda x: tf.compat.v1.py_func(load_valid_data, [x], \
                                    [tf.uint8, tf.float32, tf.float32, tf.float32, tf.uint8, tf.uint8, tf.uint8, tf.uint8]))
                                    #msa       SS          maps        strs        labels (dist, theta, phi, omega)
            iterator = tf.data.Iterator.from_structure(trainset.output_types, trainset.output_shapes)
            self.msa_in, self.SS_in, self.map_in, self.tr_crd_in, self.dist_in, self.theta_in, self.phi_in, self.omega_in = iterator.get_next()
            self.train_init_op = iterator.make_initializer(trainset)
            self.valid_init_op = iterator.make_initializer(validset)
            self.n_train = len(train_pdbs)
            self.n_valid = len(valid_pdbs)
        #
        with tf.variable_scope("trRefine") as scope:
            self.build_graph(pdb_in=pdb_in, train=train)

    def build_graph(self, pdb_in=False, train=False):
        activation = tf.nn.elu
        inst_norm = tf.contrib.layers.instance_norm
        conv1d = tf.layers.conv1d
        conv2d = tf.layers.conv2d
        #
        with tf.name_scope('input'):
            # Training parameters
            self.isTraining = tf.placeholder_with_default(False, shape=()) 
            
            if not train:
                # 1D features
                self.msa_in = tf.placeholder(tf.uint8, shape=[None, None]) # input MSA
                self.SS_in = tf.placeholder(tf.float32, shape=[None, 9]) # SS prediction results (from Emin), (n_res, 9)
                
                # 2D features
                # option1: from npz files (for training)
                self.map_in = tf.placeholder(tf.float32, shape=[None, None, None, self.crd_size]) # n_map, n_res, n_res, n_crd_feat
                self.tr_crd_in = tf.placeholder(tf.float32, shape=[None, None, None, self.str_size]) # n_map, n_res, n_res, n_crd_feat
                
                # option2: iter0 prediction from npz files (self.tr_crd_in)
                #          get N, CA, C coords from pdb and make 6D crd distrib
                self.N  = tf.placeholder(tf.float32, shape=[None, None, 3]) # n_str, n_res, xyz
                self.CA = tf.placeholder(tf.float32, shape=[None, None, 3]) # n_str, n_res, xyz
                self.C  = tf.placeholder(tf.float32, shape=[None, None, 3]) # n_str, n_res, xyz
                self.esto = tf.placeholder(tf.float32, shape=[None, None, None, 15]) # n_str, n_res, n_res, 15
                self.seqsep = tf.placeholder(tf.float32, shape=[None, None]) # n_res, n_res
                
                # Target values - For training
                self.dist_in = tf.placeholder(tf.uint8, shape=[None, None]) # n_res, n_res, 37
                self.theta_in = tf.placeholder(tf.uint8, shape=[None, None]) # n_res, n_res, 25
                self.omega_in = tf.placeholder(tf.uint8, shape=[None, None]) # n_res, n_res, 25
                self.phi_in = tf.placeholder(tf.uint8, shape=[None, None]) # n_res, n_res, 13
        
        nres = tf.shape(self.map_in)[1]

        if pdb_in:
            nmap = tf.shape(self.map_in)[0]
            transf6d = self.calc_6d_transforms(self.N, self.CA, self.C)
            #
            nstr = tf.shape(self.N)[0]
            seqsep = tf.tile(self.seqsep[None,:,:,None], [nstr, 1, 1, 1])
            str_in = tf.concat((transf6d, self.esto, seqsep), axis=-1)
            #
            map_in = self.map_in
        else:
            nmap = tf.shape(self.map_in)[0]
            nstr = tf.shape(self.tr_crd_in)[0]
            map_in = self.map_in
            str_in = self.tr_crd_in

        # convert inputs
        # 1. collect sequence features
        msa1hot = tf.one_hot(self.msa_in, N_AA_MSA, dtype=tf.float32)
        
        w = reweight_seq(msa1hot, WMIN)
        neff = tf.reduce_sum(w)
        f1d_seq = msa1hot[0, :, :20]
        f1d_pssm = msa2pssm(msa1hot, w)
        f1d = tf.concat([f1d_seq, f1d_pssm, self.SS_in], axis=-1) # (n_res, 20+22)
        f1d = tf.expand_dims(f1d, axis=0)
        f1d = tf.reshape(f1d, (1,nres,51))
        f1d = conv1d(f1d, 20, 1, padding='SAME') 

        tiled_f1d = tf.tile(f1d[:,:,None,:], [1, 1, nres, 1])

        # 2. process input 6D coords prediction
        f1d = tf.tile(tiled_f1d, [nmap, 1, 1, 1])
        t2d = tf.concat([f1d,\
                         tf.transpose(f1d, (0, 2, 1, 3)),\
                         map_in], axis=-1)
        t2d = tf.reshape(t2d, [nmap, nres, nres, 20*2+self.crd_size])
        # single block of ResNet
        t2d = build_resnet(t2d, self.half_filters, 1, self.isTraining, p_dropout=self.p_dropout)  
        t2d = activation(inst_norm(t2d))

        # self-attention
        signal = activation(conv2d(t2d, self.half_filters, 3, padding="SAME"))
        prob = tf.nn.softmax(conv2d(t2d, 1, 3, padding="SAME"), axis=0) # (nmap, nres, nres, 1)
        t2d = tf.reduce_sum(signal * prob, axis=0) # (nres, nres, 64)
        
        # 3. process input structures
        f1d = tf.tile(tiled_f1d, [nstr, 1, 1, 1])
        s2d = tf.concat([f1d,
                         tf.transpose(f1d, (0, 2, 1, 3)),
                         str_in], axis=-1)
        s2d = tf.reshape(s2d, [nstr, nres, nres, 20*2+self.str_size])
        # single block of ResNet
        s2d = build_resnet(s2d, self.half_filters, 1, self.isTraining, p_dropout=self.p_dropout)  
        s2d = activation(inst_norm(s2d))
        #
        # self-attention
        signal = activation(conv2d(s2d, self.half_filters, 3, padding="SAME"))
        prob = tf.nn.softmax(conv2d(s2d, 1, 3, padding="SAME"), axis=0) # (nmap, nres, nres, 1)
        s2d = tf.reduce_sum(signal * prob, axis=0) # (nres, nres, 64)

        feat = tf.concat([t2d[None,:,:,:], s2d[None,:,:,:]], axis=-1)
        feat = build_resnet(feat, self.n2d_filters, self.num_chunks, self.isTraining, p_dropout=self.p_dropout)
        feat = activation(inst_norm(feat))

        # Self-attention to all?
        if self.use_attention:
            feat = pixelSelfAttention(feat, maxpool=9)

        # Final convolution to predict 6D coords.
        logits_theta = build_resnet(feat, self.half_filters, 1, self.isTraining, p_dropout=self.p_dropout, final_channel=N_BIN360, inst_norm=False)
        prob_theta = tf.nn.softmax(logits_theta)
        
        logits_phi = build_resnet(feat, self.half_filters, 1, self.isTraining, p_dropout=self.p_dropout, final_channel=N_BIN180,inst_norm=False)
        prob_phi = tf.nn.softmax(logits_phi)

        # symmetrize features (dist and omega are symmetric)
        feat = 0.5 * (feat + tf.transpose(feat, (0,2,1,3)))
        logits_dist = build_resnet(feat, self.half_filters, 1, self.isTraining, p_dropout=self.p_dropout, final_channel=N_DIST_BIN, inst_norm=False)
        prob_dist = tf.nn.softmax(logits_dist)
        
        logits_omega = build_resnet(feat, self.half_filters, 1, self.isTraining, p_dropout=self.p_dropout, final_channel=N_BIN360, inst_norm=False)
        prob_omega = tf.nn.softmax(logits_omega)

        # loss functions
        dist = tf.one_hot(self.dist_in, N_DIST_BIN, dtype=tf.float32)
        dist = tf.expand_dims(dist, 0)
        theta = tf.one_hot(self.theta_in, N_BIN360, dtype=tf.float32)
        theta = tf.expand_dims(theta, 0)
        omega = tf.one_hot(self.omega_in, N_BIN360, dtype=tf.float32)
        omega = tf.expand_dims(omega, 0)
        phi = tf.one_hot(self.phi_in, N_BIN180, dtype=tf.float32)
        phi = tf.expand_dims(phi, 0)
        #
        loss_dist = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(dist, logits_dist))
        loss_theta = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(theta, logits_theta))
        loss_phi = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(phi, logits_phi))
        loss_omega = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(omega, logits_omega))

        self.t_vars = tf.trainable_variables()

        l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in self.t_vars if 'kernel' in var.name]) * self.l2_coeff

        self.loss_s = [loss_dist, loss_theta, loss_phi, loss_omega, l2_loss]
        self.tot_loss = tf.add_n(self.loss_s)

        self.prob_s = [prob_dist, prob_theta, prob_phi, prob_omega]

        # define saver
        self.saver = tf.train.Saver()
    
    def calc_distance(self, A, B, transform=False):
        # A & B: (n_conf, n_res, 3)
        # D = row_norms_A - 2*A*B + row_norms_B
        n_str = tf.shape(A)[0]
        n_res_A = tf.shape(A)[1]
        n_res_B = tf.shape(B)[1]
        #
        row_norms_A = tf.reduce_sum(tf.square(A), axis=-1)
        row_norms_A = tf.reshape(row_norms_A, [n_str, n_res_A, 1])
        #
        row_norms_B = tf.reduce_sum(tf.square(B), axis=-1)
        row_norms_B = tf.reshape(row_norms_B, [n_str, 1, n_res_B])
        #
        D = row_norms_A - 2 * tf.matmul(A, tf.transpose(B, perm=(0,2,1))) + row_norms_B # squared distance
        D = tf.maximum(D, 0.0)
        D = tf.sqrt(D) # (n_str, n_res_A, n_res_B)
        #
        if transform:
            return f_arcsinh(D)
        else:
            return D

    def get_dihedrals(self, A, B, C, D):
        if len(A.shape) > 2:
            A = tf.reshape(A, [-1, 3])
            B = tf.reshape(B, [-1, 3])
            C = tf.reshape(C, [-1, 3])
            D = tf.reshape(D, [-1, 3])
        #
        B0 = -1.0*(B - A) # n_conf*n_res, 3
        B1 = C - B
        B2 = D - C
        #
        B1 /= tf.linalg.norm(B1, axis=-1)[:,None]
        #
        V = B0 - tf.reduce_sum(B0*B1, axis=-1)[:,None]*B1
        W = B2 - tf.reduce_sum(B2*B1, axis=-1)[:,None]*B1
        #
        X = tf.reduce_sum(V*W, axis=-1)
        Y = tf.reduce_sum(tf.cross(B1, V)*W, axis=-1)
        #
        ang_raw = tf.atan2(Y, X)
        ang = tf.where(tf.is_nan(ang_raw), tf.zeros_like(ang_raw), ang_raw)
        return ang 

    def get_angles(self, A, B, C):
        if len(A.shape) > 2:
            A = tf.reshape(A, [-1, 3])
            B = tf.reshape(B, [-1, 3])
            C = tf.reshape(C, [-1, 3])
        V = A - B
        V /= tf.linalg.norm(V, axis=-1)[:,None] # normalized vector

        W = C - B
        W /= tf.linalg.norm(W, axis=-1)[:,None]
        
        X = tf.reduce_sum(V*W, axis=-1) # dot product v dot w = |v||w|cos(theta)

        ang_raw = tf.acos(X)
        ang = tf.where(tf.is_nan(ang_raw), tf.zeros_like(ang_raw), ang_raw)
        return ang

    def get_virtual_CB(self, N, CA, C):
        b = CA - N
        c = C - CA
        a = tf.cross(b, c)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA
        return Cb
    
    def calc_6d_transforms(self, N, CA, C, d_max=20.0):
        Cb = self.get_virtual_CB(N, CA, C)
        # calc 6D transforms for all pairs within d_max
        dist = self.calc_distance(Cb, Cb)
        idx = tf.where(tf.logical_and(dist < d_max, dist > 0.0)) # [# of true, 3], 0 for batch, 1 for res_1, 2 for res_2
        #
        idx_1 = idx[:,:2] # (N, 2)
        idx_2 = tf.stack((idx[:,0], idx[:,2]), axis=-1)
        #
        CA_1 = tf.gather_nd(CA, idx_1)
        CA_2 = tf.gather_nd(CA, idx_2)
        #
        Cb_1 = tf.gather_nd(Cb, idx_1)
        Cb_2 = tf.gather_nd(Cb, idx_2)
        #
        N_1 = tf.gather_nd(N, idx_1)
        N_2 = tf.gather_nd(N, idx_2)
        # 
        #omega6d = tf.zeros_like(dist) # (n_str, n_res, n_res)
        #theta6d = tf.zeros_like(dist)
        #phi6d = tf.zeros_like(dist)
        #
        ang = self.get_dihedrals(CA_1, Cb_1, Cb_2, CA_2)
        omega6d = tf.scatter_nd(idx, ang, tf.shape(dist, out_type=tf.int64))
        #
        ang = self.get_dihedrals(N_1, CA_1, Cb_1, Cb_2)
        theta6d = tf.scatter_nd(idx, ang, tf.shape(dist, out_type=tf.int64))
        #
        ang = self.get_angles(CA_1, Cb_1, Cb_2)
        phi6d = tf.scatter_nd(idx, ang, tf.shape(dist, out_type=tf.int64))
        #
        mask6d = tf.zeros_like(dist)
        mask6d = tf.scatter_nd(idx, tf.ones_like(ang), tf.shape(mask6d, out_type=tf.int64))
        #
        orien = tf.stack((omega6d*mask6d, theta6d*mask6d, phi6d*mask6d), axis=-1)
        orien = tf.concat((tf.sin(orien), tf.cos(orien)), axis=-1)
        orien = orien * mask6d[:,:,:,None]
        #return dist, omega6d, theta6d, phi6d
        #
        # transf
        dist = f_arcsinh(dist)
        return tf.concat((dist[:,:,:,None]*mask6d[:,:,:,None], orien), axis=-1)

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

    def train(self, config_file):
        with open(config_file) as json_file:
            config = json.load(json_file, object_hook=Json_param)
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
                                            decay_steps=self.n_train*config.decay_step,
                                            decay_rate=0.5,
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
        ops_to_run = [optim, lr, self.tot_loss, self.loss_s]
        ops_to_val = [self.tot_loss, self.loss_s]

        # initialize all variables
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self.sess.run(init_op)
        #
        counter = 0
        self.start_time = time.time()
        #
        min_val_loss = config.best_loss
        #
        # Try to load pre-trained model if exists
        could_load = self.load("model_%d_%d"%(self.num_chunks, self.n2d_filters), 'last_epoch')
        if not could_load:
            could_load = self.load("model_%d_%d"%(self.num_chunks, self.n2d_filters), 'model')
        #
        for epoch in range(config.n_epoch):
            tot_loss_value = 0.0
            tot_loss_s = np.zeros(5, dtype=np.float32)
            n_tot = 0.0
            #
            # initialize trainset iterator
            self.sess.run(self.train_init_op)
            for i in range(self.n_train):
                #TODO
                _, decayed_lr, loss_value, loss_s = self.sess.run(ops_to_run,
                                         feed_dict={
                                             self.isTraining: True})
                tot_loss_value += loss_value
                tot_loss_s += np.array(loss_s)
                n_tot += 1.0
                #
                counter += 1
                if counter % N_PRINT_LEVEL == 0:
                    loss_value = tot_loss_value/n_tot
                    loss_s = tot_loss_s/n_tot
                    tot_loss_value = 0.0
                    tot_loss_s = np.zeros(5, dtype=np.float32)
                    n_tot = 0.0
                    log_list = [epoch, config.n_epoch, counter, time.time()-self.start_time, decayed_lr*100.0, loss_value-loss_s[-1]]
                    log_list.extend(loss_s)
                    print (TRAIN_LOG%tuple(log_list))
            #
            tot_loss_value = 0.0
            tot_loss_s = np.zeros(5, dtype=np.float32)
            n_tot = 0.0
            #
            # initialize validation iterator
            self.sess.run(self.valid_init_op)
            for i in range(self.n_valid):
                loss_value, loss_s = self.sess.run(ops_to_val,
                                         feed_dict={
                                             self.isTraining: False})
                tot_loss_value += loss_value
                tot_loss_s += np.array(loss_s)
                n_tot += 1.0
            #
            loss_value = tot_loss_value/n_tot
            loss_s = tot_loss_s/n_tot
            #
            log_list = [epoch, config.n_epoch, counter, time.time()-self.start_time, decayed_lr*100.0, loss_value-loss_s[-1]]
            log_list.extend(loss_s)
            print (VALID_LOG%tuple(log_list))
            val_loss = loss_value - loss_s[-1]
            if val_loss < min_val_loss:
                self.save("model_%d_%d"%(self.num_chunks, self.n2d_filters), 'model')
                min_val_loss = val_loss
            self.save("model_%d_%d"%(self.num_chunks, self.n2d_filters), 'last_epoch')

    def test(self):
        return

    def predict(self, model_dir, a3m_fn, npz_fn_s, pdb_fn_s, acc_fn_s, SS_fn, out_fn):
        if os.path.exists(out_fn):
            return

        # load model
        import glob
        model_dir_s = glob.glob("%s_rep?"%(model_dir))
        #
        msa, SS, N, CA, C, esto, seqsep, maps = process_inputs(a3m_fn, npz_fn_s, pdb_fn_s, acc_fn_s, SS_fn)
        nres = len(msa[0])
        dist = np.zeros((nres, nres, N_DIST_BIN))
        theta = np.zeros((nres, nres, N_BIN360))
        omega = np.zeros((nres, nres, N_BIN360))
        phi = np.zeros((nres, nres, N_BIN180))
        for model_dir in model_dir_s:
            self.load(model_dir, 'model')
            prob_s = self.sess.run(self.prob_s,
                                   feed_dict={
                                       self.msa_in: msa,
                                       self.SS_in: SS,
                                       self.N: N,
                                       self.CA: CA, 
                                       self.C: C,
                                       self.esto: esto,
                                       self.seqsep: seqsep,
                                       self.map_in: maps,
                                       self.isTraining: False})
            dist += prob_s[0][0]
            theta += prob_s[1][0]
            phi   += prob_s[2][0]
            omega += prob_s[3][0]
        dist = dist / float(len(model_dir_s))
        theta = theta / float(len(model_dir_s))
        phi = phi / float(len(model_dir_s))
        omega = omega / float(len(model_dir_s))
        #
        # save to npz file
        np.savez_compressed(out_fn, dist=dist.astype(np.float16),\
                                    theta=theta.astype(np.float16),\
                                    phi=phi.astype(np.float16),\
                                    omega=omega.astype(np.float16))
