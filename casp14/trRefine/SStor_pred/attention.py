#!/usr/bin/env python

import tensorflow as tf

def attention(inputs, attention_size, time_major=False, return_alphas=False):
    if isinstance(inputs, tuple):
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T, B, D) => (B, T, D)
        inputs = tf.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value # hidden size of the RNN layer

    # trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1), name='kernel')
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name='bias')
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1), name='kernel')

    with tf.name_scope('v'):
        # applying fully connected layer with non-linear activation to each of the B*T timestamps
        # the shape of v is (B, T, D)*(D, A) = (B, T, A)
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # for each of the timesteps its vector of size A from 'v' is reduced with 'u' vector
    # the shape of vu is (B,T,A)*(A) = (B, T)
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')
    alphas = tf.nn.softmax(vu, name='alphas')

    # output: (B, T, D)*(B, T, *) = (B, T, D) -> sum over T => (B, D)
    output = tf.reduce_sum(inputs*tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    return output, alphas
