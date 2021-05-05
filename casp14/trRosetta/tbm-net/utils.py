import os
import tensorflow as tf
from collections import namedtuple

FFindexDB = namedtuple("FFindexDB", "index, data")

def load_weights_no1d(args, params):

    w,b = [],[]
    beta,gamma = [],[]

    nconv2d  = (params['CYCLS1']+params['CYCLS2'])*len(params['DRATES'])*2+10
    nlaynorm = nconv2d-7

    for filename in os.listdir(args.MDIR):
        if not filename.endswith(".index"):
            continue
        mname = args.MDIR+"/"+os.path.splitext(filename)[0]
        print('reading weights from:', mname)

        w.append([
            tf.train.load_variable(mname, 'conv2d/kernel')
            if i==0 else
            tf.train.load_variable(mname, 'conv2d_%d/kernel'%i)
            for i in range(nconv2d)])

        b.append([
            tf.train.load_variable(mname, 'conv2d/bias')
            if i==0 else
            tf.train.load_variable(mname, 'conv2d_%d/bias'%i)
            for i in range(nconv2d)])

        beta.append([
            tf.train.load_variable(mname, 'InstanceNorm/beta')
            if i==0 else
            tf.train.load_variable(mname, 'InstanceNorm_%d/beta'%i)
            for i in range(nlaynorm)])

        gamma.append([
            tf.train.load_variable(mname, 'InstanceNorm/gamma')
            if i==0 else
            tf.train.load_variable(mname, 'InstanceNorm_%d/gamma'%i)
            for i in range(nlaynorm)])

    return (w,b,beta,gamma)


def load_weights(args, params):

    w,b = [],[]
    beta,gamma = [],[]
    w1d,b1d = [],[]

    nconv2d  = (params['CYCLS1']+params['CYCLS2'])*len(params['DRATES'])*2+10
    nlaynorm = nconv2d-7

    for filename in os.listdir(args.MDIR):
        if not filename.endswith(".index"):
            continue
        mname = args.MDIR+"/"+os.path.splitext(filename)[0]
        print('reading weights from:', mname)

        w1d.append(tf.train.load_variable(mname, 'conv1d/kernel'))
        b1d.append(tf.train.load_variable(mname, 'conv1d/bias'))

        w.append([
            tf.train.load_variable(mname, 'conv2d/kernel')
            if i==0 else
            tf.train.load_variable(mname, 'conv2d_%d/kernel'%i)
            for i in range(nconv2d)])

        b.append([
            tf.train.load_variable(mname, 'conv2d/bias')
            if i==0 else
            tf.train.load_variable(mname, 'conv2d_%d/bias'%i)
            for i in range(nconv2d)])

        beta.append([
            tf.train.load_variable(mname, 'InstanceNorm/beta')
            if i==0 else
            tf.train.load_variable(mname, 'InstanceNorm_%d/beta'%i)
            for i in range(nlaynorm)])

        gamma.append([
            tf.train.load_variable(mname, 'InstanceNorm/gamma')
            if i==0 else
            tf.train.load_variable(mname, 'InstanceNorm_%d/gamma'%i)
            for i in range(nlaynorm)])

    return (w,b,beta,gamma,w1d,b1d)


def InstanceNorm(features,beta,gamma):
    mean,var = tf.nn.moments(features,axes=[1,2])
    x = (features - mean[:,None,None,:]) / tf.sqrt(var[:,None,None,:]+1e-5)
    out = tf.constant(gamma)[None,None,None,:]*x + tf.constant(beta)[None,None,None,:]
    return out

def Conv2d(features,w,b,d=1):
    x = tf.nn.conv2d(features,tf.constant(w),strides=[1,1,1,1],padding="SAME",dilations=[1,d,d,1]) + tf.constant(b)[None,None,None,:]
    return x

def Conv1d(features,w,b,d=1):
    x = tf.nn.conv1d(features,tf.constant(w),stride=[1,1,1],padding="SAME",dilations=[1,d,1]) + tf.constant(b)[None,None,:]
    return x

Activation = tf.nn.elu

def resblock(layers2d,w,b,beta,gamma,dilation,i,j):
    layers2d.append(Conv2d(layers2d[-1],w[i],b[i],dilation))
    layers2d.append(InstanceNorm(layers2d[-1],beta[j],gamma[j]))
    layers2d.append(Activation(layers2d[-1]))
    layers2d.append(Conv2d(layers2d[-1],w[i+1],b[i+1],dilation))
    layers2d.append(InstanceNorm(layers2d[-1],beta[j+1],gamma[j+1]))
    layers2d.append(Activation(layers2d[-1] + layers2d[-6]))
