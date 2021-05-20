#!/usr/bin/env python3

import tensorflow as tf

class Json_param:
    def __init__(self, dict):
        vars(self).update(dict)

def batch_norm(inputs, training):
    return tf.layers.batch_normalization(inputs, training=training)

def inst_norm(inputs):
    return tf.contrib.layers.instance_norm(inputs)

def reweight_seq(msa1hot, cutoff):
    with tf.variable_scope('reweight'):
        id_min = tf.cast(tf.shape(msa1hot)[1], tf.float32) * cutoff # msa1hot.shape[1] == n_res
        id_mtx = tf.tensordot(msa1hot, msa1hot, [[1,2], [1,2]])
        id_mask = id_mtx > id_min
        w = 1.0/tf.reduce_sum(tf.cast(id_mask, dtype=tf.float32),-1)
    return w

def msa2pssm(msa1hot, w):
    beff = tf.reduce_sum(w)
    f_i = tf.reduce_sum(w[:,None,None]*msa1hot, axis=0) / beff + 1e-9
    h_i = tf.reduce_sum( -f_i * tf.log(f_i+1e-9), axis=1)    
    return tf.concat([f_i, h_i[:,None]], axis=1)

# for mixed precision training
def float16_variable_storage_getter(getter, name, shape=None, dtype=tf.float16,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    storage_dtype = tf.float16 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float16:
        variable = tf.cast(variable, dtype)
    return variable

def float32_variable_storage_getter(getter, name, shape=None, dtype=tf.float16,
                                    initializer=None, regularizer=None,
                                    trainable=True,
                                    *args, **kwargs):
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable

def gradients_with_loss_scaling(loss, variables, loss_scale):
    return [grad / loss_scale for grad in tf.gradients(loss*loss_scale, variables)]

