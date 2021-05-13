import os
import json
import numpy as np
import tensorflow as tf
from features import get_features

def load_weights(MDIR):
    '''load trRosetta weights from all checkpoints 
    in a given folder'''

    with open(MDIR + '/params.json') as jsonfile:
        params = json.load(jsonfile)
    
    w,b = [],[]
    beta,gamma = [],[]
    w1d,b1d = [],[]

    if params['NETTYPE'] == 'TBM':
        nconv2d  = (params['CYCLS1']+params['CYCLS2'])*len(params['DRATES'])*2+10
        nlaynorm = nconv2d-7
    elif params['NETTYPE'] == 'MSA':
        nconv2d  = (params['CYCLS1']+params['CYCLS2'])*len(params['DRATES'])*2+8
        nlaynorm = nconv2d-5
    else:
        sys.exit("Wrong network type: '%s'"%(params['NETTYPE']))
        
    for filename in os.listdir(MDIR):
        if not filename.endswith(".index"):
            continue
        mname = MDIR+"/"+os.path.splitext(filename)[0]
        print('loading weights from:', mname)

        #if params['NETTYPE'] == 'MSA':
        if 'conv1d/bias' in [v[0] for v in tf.train.list_variables(mname)]:
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

    return (w,b,beta,gamma,w1d,b1d),params


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

    
def build_msa_net(msa,ins,tape,idx, weights, params):
    '''sequence-based trRosetta'''
    w,b,beta,gamma,w1d,b1d = weights

    # extract features from an MSA
    ncol = tf.shape(msa)[1]
    msa_features = get_features(msa,ins,idx,params)

    # network
    layers2d = [[] for _ in range(len(w))]
    preds = [[] for _ in range(4)] # theta,phi,dist,omega

    # create a separate branch for every checkpoint
    for i in range(len(w)):

        # project down TAPE features
        tape_reduced = Activation(Conv1d(tape, w1d[i], b1d[i]))

        # join MSA and TAPE features
        inputs = tf.concat([msa_features,
                            tf.tile(tape_reduced[:,:,None,:], [1,1,ncol,1]),
                            tf.tile(tape_reduced[:,None,:,:], [1,ncol,1,1])], axis=-1)

        # project features down
        layers2d[i] = [inputs]
        layers2d[i].append(Conv2d(layers2d[i][-1],w[i][0],b[i][0]))
        layers2d[i].append(InstanceNorm(layers2d[i][-1],beta[i][0],gamma[i][0]))
        layers2d[i].append(Activation(layers2d[i][-1]))

        # first cycle with more filters
        k = 1
        for _ in range(params['CYCLS1']):
            for dilation in params['DRATES']:
                resblock(layers2d[i],w[i],b[i],beta[i],gamma[i],dilation,k,k)
                k += 2

        # project down
        layers2d[i].append(Conv2d(layers2d[i][-1],w[i][k],b[i][k],1))
        layers2d[i].append(Activation(layers2d[i][-1]))
        k += 1

        # second cycle with less filters
        for _ in range(params['CYCLS2']):
            for dilation in params['DRATES']:
                resblock(layers2d[i],w[i],b[i],beta[i],gamma[i],dilation,k,k-1)
                k += 2

        # one more block with dilation=1
        resblock(layers2d[i],w[i],b[i],beta[i],gamma[i],1,k,k-1)

        # probabilities for theta and phi
        preds[0].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][-4],b[i][-4]))[0])
        preds[1].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][-3],b[i][-3]))[0])

        # symmetrize
        layers2d[i].append(0.5*(layers2d[i][-1]+tf.transpose(layers2d[i][-1],perm=[0,2,1,3])))

        # probabilities for dist and omega
        preds[2].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][-2],b[i][-2]))[0])
        preds[3].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][-1],b[i][-1]))[0])

    # average over all branches
    pt = tf.reduce_mean(tf.stack(preds[0]),axis=0)
    pp = tf.reduce_mean(tf.stack(preds[1]),axis=0)
    pd = tf.reduce_mean(tf.stack(preds[2]),axis=0)
    po = tf.reduce_mean(tf.stack(preds[3]),axis=0)

    return pd,po,pt,pp


def build_tbm_net(msa,ins,tape,idx,t1d,t2d, weights, params):
    '''template-based trRosetta'''
    w,b,beta,gamma,w1d,b1d = weights

    # extract features from an MSA
    ncol = tf.shape(msa)[1]
    msa_features = get_features(msa,ins,idx,params)

    # prepare template features for processing
    t1d2d = tf.concat([tf.one_hot(tf.cast(t2d[:,:,:,0],dtype=tf.uint8),params['DBINS']+1,dtype=tf.float32),
                        tf.sin(t2d[:,:,:,1])[:,:,:,None],
                        tf.cos(t2d[:,:,:,1])[:,:,:,None],
                        tf.sin(t2d[:,:,:,2])[:,:,:,None],
                        tf.cos(t2d[:,:,:,2])[:,:,:,None],
                        tf.sin(t2d[:,:,:,3])[:,:,:,None],
                        tf.cos(t2d[:,:,:,3])[:,:,:,None],
                        t2d[:,:,:,4:],
                        tf.tile(t1d[:,None,:,768:],[1,ncol,1,1]),
                        tf.tile(t1d[:,:,None,768:],[1,1,ncol,1])],axis=-1)

    # lists to store separate branches and their predictions
    layers2d = [[] for _ in range(len(w))]
    preds = [[] for _ in range(4)] # theta,phi,dist,omega

    # create a separate branch for every checkpoint
    for i in range(len(w)):

        # process template features
        sgnl = Activation(Conv2d(t1d2d,w[i][0],b[i][0]))
        prob = tf.nn.softmax(Conv2d(t2d,w[i][1],b[i][1]), axis=0)
        inputs_tmp = tf.reduce_sum(sgnl*prob,axis=0)

        fin = [inputs_tmp[None,:,:,:], msa_features]

        if len(w1d)>0:
            # add TAPE features if there are weights to process them
            tape_reduced = Activation(Conv1d(tape, w1d[i], b1d[i]))
            fin += [tf.tile(tape_reduced[:,:,None,:], [1,1,ncol,1]),
                    tf.tile(tape_reduced[:,None,:,:], [1,ncol,1,1])]

        inputs = tf.concat(fin, axis=-1)

        # project features down
        layers2d[i] = [inputs]
        layers2d[i].append(Conv2d(layers2d[i][-1],w[i][2],b[i][2]))
        layers2d[i].append(InstanceNorm(layers2d[i][-1],beta[i][0],gamma[i][0]))
        layers2d[i].append(Activation(layers2d[i][-1]))

        # first cycle with more filters
        k = 1
        for _ in range(params['CYCLS1']):
            for dilation in params['DRATES']:
                resblock(layers2d[i],w[i],b[i],beta[i],gamma[i],dilation,k+2,k)
                k += 2

        # project down
        layers2d[i].append(Conv2d(layers2d[i][-1],w[i][k+2],b[i][k+2],1))
        layers2d[i].append(Activation(layers2d[i][-1]))
        k += 1

        # second cycle with less filters
        for _ in range(params['CYCLS2']):
            for dilation in params['DRATES']:
                resblock(layers2d[i],w[i],b[i],beta[i],gamma[i],dilation,k+2,k-1)
                k += 2

        # one more block with dilation=1
        resblock(layers2d[i],w[i],b[i],beta[i],gamma[i],1,k+2,k-1)

        # probabilities for theta and phi
        preds[0].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][-4],b[i][-4]))[0])
        preds[1].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][-3],b[i][-3]))[0])

        # symmetrize
        layers2d[i].append(0.5*(layers2d[i][-1]+tf.transpose(layers2d[i][-1],perm=[0,2,1,3])))

        # probabilities for dist and omega
        preds[2].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][-2],b[i][-2]))[0])
        preds[3].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][-1],b[i][-1]))[0])

    # average over all branches
    pt = tf.reduce_mean(tf.stack(preds[0]),axis=0)
    pp = tf.reduce_mean(tf.stack(preds[1]),axis=0)
    pd = tf.reduce_mean(tf.stack(preds[2]),axis=0)
    po = tf.reduce_mean(tf.stack(preds[3]),axis=0)

    return pd,po,pt,pp


def get_cont_crop(inputs,i,window,cov,maxseq):
    '''continuous cropping'''
    
    j = i+window
    msa_ = inputs['msa'][:,i:j]
    ins_ = inputs['ins'][:,i:j]
    mask = np.sum(msa_==20,axis=-1)<(cov*window)
    msa_ = msa_[mask][:maxseq]
    ins_ = ins_[mask][:maxseq]

    print("window_%d: pos=%d N(seq)=%d"%(window,i,msa_.shape[0]))

    feed_dict = {
        'msa'  : msa_,
        'ins'  : ins_,
        'tape' : inputs['tape'][:,i:j],
        'idx'  : np.arange(i,j)
    }

    if 't1d' in inputs.keys():
        feed_dict.update({
            't1d' : inputs['t1d'][:,i:j],
            't2d' : inputs['t2d'][:,i:j,i:j],
        })
        
    return feed_dict


def get_discont_crop(inputs,i,j,wi,wj,cov,maxseq):
    '''discontinuous cropping'''
    
    L = inputs['msa'].shape[1]
    sel = np.zeros((L)).astype(np.bool)
    sel[i:i+wi] = True
    sel[j:j+wj] = True

    msa_ = inputs['msa'][:,sel]
    ins_ = inputs['ins'][:,sel]
    mask = np.sum(msa_==20,axis=-1)<(cov*wi)
    msa_ = msa_[mask][:maxseq]
    ins_ = ins_[mask][:maxseq]
    idx_ = np.arange(L)[sel]

    print("window_%dx%d: pos=(%d,%d) N(seq)=%d"%(wi,wj,i,j,msa_.shape[0]))
    
    feed_dict = {
        'msa'  : msa_,
        'ins'  : ins_,
        'tape' : inputs['tape'][:,sel],
        'idx'  : idx_
    }

    if 't1d' in inputs.keys():
        feed_dict.update({
            't1d' : inputs['t1d'][:,sel],
            't2d' : inputs['t2d'][:,sel][:,:,sel],
        })
        
    return feed_dict,idx_


def get_mask2d(L1,L2):

    idx1 = np.arange(L1//2)
    if L1%2==0:
        idx1 = np.concatenate([idx1,idx1[::-1]])
    else:
        idx1 = np.concatenate([idx1,[L1//2],idx1[::-1]])

    idx2 = np.arange(L2//2)
    if L2%2==0:
        idx2 = np.concatenate([idx2,idx2[::-1]])
    else:
        idx2 = np.concatenate([idx2,[L2//2],idx2[::-1]])

    idx = np.concatenate([idx1,idx2])
    L = np.minimum(L1,L2)

    idx2d = np.minimum(idx[None,:],idx[:,None])+1
    idx2d[idx2d>L//3] = L//3
    return idx2d
