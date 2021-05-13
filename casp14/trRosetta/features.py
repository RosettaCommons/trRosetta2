import numpy as np
import tensorflow as tf

# reweight MSA based on cutoff
def reweight(msa1hot, cutoff):

    with tf.name_scope('reweight'):
        id_min = tf.cast(tf.shape(msa1hot)[1], tf.float32) * cutoff
        id_mtx = tf.tensordot(msa1hot, msa1hot, [[1,2], [1,2]])
        id_mask = id_mtx > id_min
        w = 1.0/tf.reduce_sum(tf.cast(id_mask, dtype=tf.float32),-1)
    return w

# 1d features
def get_features1d(msa1hot,insertions,w):
    nc = tf.shape(msa1hot)[1]
    beff = tf.reduce_sum(w)
    msw = w[:,None,None]*msa1hot
    f_i = tf.reduce_sum(msw, axis=0) / beff + 1e-9
    h_i = tf.reduce_sum( -f_i * tf.math.log(f_i), axis=1)
    n_i = tf.math.log(tf.reduce_sum(msw[:,:,:20],axis=[0,2]))
    ins = tf.reduce_mean(tf.cast(insertions>0,dtype=tf.float32),axis=0)
    f1d = tf.concat([msa1hot[0,:,:20], f_i, h_i[:,None], n_i[:,None], ins[:,None]], axis=1)
    f1d = tf.reshape(f1d, [1,nc,44])
    return f1d


# 2d features
def get_features2d(msa1hot, weights, penalty):

    nr = tf.shape(msa1hot)[0]
    nc = tf.shape(msa1hot)[1]
    ns = tf.shape(msa1hot)[2]

    neff = tf.reduce_sum(weights)

    with tf.name_scope('covariance'):
        x = tf.reshape(msa1hot, (nr, nc * ns))
        num_points = tf.reduce_sum(weights) - tf.sqrt(tf.reduce_mean(weights))
        mean = tf.reduce_sum(x * weights[:,None], axis=0, keepdims=True) / num_points
        x = (x - mean) * tf.sqrt(weights[:,None])
        cov = tf.matmul(tf.transpose(x), x)/num_points

    with tf.name_scope('inv_convariance'):
        cov_reg = cov + tf.eye(nc * ns) * penalty / tf.sqrt(neff)
        inv_cov = tf.linalg.inv(cov_reg)

        x1 = tf.reshape(inv_cov,(nc, ns, nc, ns))
        x2 = tf.transpose(x1, [0,2,1,3])
        features = tf.reshape(x2, (nc, nc, ns * ns))

        x3 = tf.sqrt(tf.reduce_sum(tf.square(x1[:,:-1,:,:-1]),(1,3))) * (1-tf.eye(nc))
        apc = tf.reduce_sum(x3,0,keepdims=True) * tf.reduce_sum(x3,1,keepdims=True) / tf.reduce_sum(x3)
        contacts = (x3 - apc) * (1-tf.eye(nc))

    with tf.name_scope('misc_2d'):
        f_ij = tf.tensordot(weights[:,None,None]*msa1hot, msa1hot, [[0],[0]]) / neff + 1e-9
        gaps = f_ij[:,20,:,20]
        h_ij = tf.reduce_sum( -f_ij * tf.math.log(f_ij), axis=[1,3])

    f2d = tf.concat([features, contacts[:,:,None], h_ij[:,:,None], gaps[:,:,None]], axis=2)
    f2d = tf.reshape(f2d, [1,nc,nc,444])

    return f2d


# network inputs
def get_features(msa, ins, idx, params):

    nrow = tf.shape(msa)[0]
    ncol = tf.shape(msa)[1]

    # convert msa to 1-hot
    msa1hot  = tf.one_hot(msa, 21, dtype=tf.float32)

    # extract 1d and 2d features from the msa
    w = reweight(msa1hot, params['WMIN'])
    f1d = get_features1d(msa1hot,ins,w)
    f2d = tf.cond(nrow>1, lambda: get_features2d(msa1hot, w, params['DCAREG']),
            lambda: tf.zeros([1,ncol,ncol,444], tf.float32))

    # get sequence separation
    seqsep = tf.abs(idx[:,None]-idx[None,:])+1
    seqsep = tf.math.log(tf.cast(seqsep,dtype=tf.float32))

    # stack all features together
    inputs = tf.concat([tf.tile(f1d[:,:,None,:], [1,1,ncol,1]),
                        tf.tile(f1d[:,None,:,:], [1,ncol,1,1]),
                        f2d,
                        seqsep[None,:,:,None]], axis=-1)
    inputs = tf.reshape(inputs, [1,ncol,ncol,2*44+444+1])

    return inputs
