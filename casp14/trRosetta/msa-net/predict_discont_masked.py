import tensorflow as tf
import numpy as np
import string
import sys,os
import json
import time
from features import get_features
from parsers import parse_a3m
from arguments import get_args

config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.90)
)

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


args = get_args()

#scriptdir = os.path.dirname(os.path.realpath(__file__))
with open(args.MDIR + '/params.json') as jsonfile:
    params = json.load(jsonfile)

a3m_,ins_ = parse_a3m(args.MSA)
a3m_ = a3m_[:5000]
ins_ = ins_[:5000]

L = a3m_.shape[1]
idx_ = np.arange(L)

tape_ = np.load(args.TAPE)[:,1:-1,:]

frac = 0.5

w,b = [],[]
beta,gamma = [],[]
w1d,b1d = [],[]

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
        for i in range(78)])

    b.append([
        tf.train.load_variable(mname, 'conv2d/bias')
        if i==0 else
        tf.train.load_variable(mname, 'conv2d_%d/bias'%i)
        for i in range(78)])

    beta.append([
        tf.train.load_variable(mname, 'InstanceNorm/beta')
        if i==0 else
        tf.train.load_variable(mname, 'InstanceNorm_%d/beta'%i)
        for i in range(73)])

    gamma.append([
        tf.train.load_variable(mname, 'InstanceNorm/gamma')
        if i==0 else
        tf.train.load_variable(mname, 'InstanceNorm_%d/gamma'%i)
        for i in range(73)])


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

with tf.Graph().as_default():

    with tf.name_scope('input'):
        msa = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None,None))
        ins = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None,None))
        tape = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1,None,768))
        idx = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None))

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
                layers2d[i].append(Conv2d(layers2d[i][-1],w[i][k],b[i][k],dilation))
                layers2d[i].append(InstanceNorm(layers2d[i][-1],beta[i][k],gamma[i][k]))
                layers2d[i].append(Activation(layers2d[i][-1]))
                k += 1
                layers2d[i].append(Conv2d(layers2d[i][-1],w[i][k],b[i][k],dilation))
                layers2d[i].append(InstanceNorm(layers2d[i][-1],beta[i][k],gamma[i][k]))
                layers2d[i].append(Activation(layers2d[i][-1] + layers2d[i][-6]))
                k += 1

        # project down
        layers2d[i].append(Conv2d(layers2d[i][-1],w[i][k],b[i][k],1))
        layers2d[i].append(Activation(layers2d[i][-1]))
        k += 1

        # second cycle with less filters
        for _ in range(params['CYCLS2']):
            for dilation in params['DRATES']:
                layers2d[i].append(Conv2d(layers2d[i][-1],w[i][k],b[i][k],dilation))
                layers2d[i].append(InstanceNorm(layers2d[i][-1],beta[i][k-1],gamma[i][k-1]))
                layers2d[i].append(Activation(layers2d[i][-1]))
                k += 1
                layers2d[i].append(Conv2d(layers2d[i][-1],w[i][k],b[i][k],dilation))
                layers2d[i].append(InstanceNorm(layers2d[i][-1],beta[i][k-1],gamma[i][k-1]))
                layers2d[i].append(Activation(layers2d[i][-1] + layers2d[i][-6]))
                k += 1

        # one more block with dilation=1
        layers2d[i].append(Conv2d(layers2d[i][-1],w[i][k],b[i][k],1))
        layers2d[i].append(InstanceNorm(layers2d[i][-1],beta[i][k-1],gamma[i][k-1]))
        layers2d[i].append(Activation(layers2d[i][-1]))
        k += 1
        layers2d[i].append(Conv2d(layers2d[i][-1],w[i][k],b[i][k],1))
        layers2d[i].append(InstanceNorm(layers2d[i][-1],beta[i][k-1],gamma[i][k-1]))
        layers2d[i].append(Activation(layers2d[i][-1] + layers2d[i][-6]))

        # probabilities for theta and phi
        preds[0].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][74],b[i][74]))[0])
        preds[1].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][75],b[i][75]))[0])

        # symmetrize
        layers2d[i].append(0.5*(layers2d[i][-1]+tf.transpose(layers2d[i][-1],perm=[0,2,1,3])))

        # probabilities for dist and omega
        preds[2].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][76],b[i][76]))[0])
        preds[3].append(tf.nn.softmax(Conv2d(layers2d[i][-1],w[i][77],b[i][77]))[0])

    # average over all branches
    pt = tf.reduce_mean(tf.stack(preds[0]),axis=0)
    pp = tf.reduce_mean(tf.stack(preds[1]),axis=0)
    pd = tf.reduce_mean(tf.stack(preds[2]),axis=0)
    po = tf.reduce_mean(tf.stack(preds[3]),axis=0)

    with tf.compat.v1.Session(config=config) as sess:

        counts = np.zeros((L,L))

        # predict using full alignment
        if L<=128:
            counts += 1
            sub_a3m = a3m_[np.sum(a3m_==20,axis=-1)<(frac*L)]
            sub_ins = ins_[np.sum(a3m_==20,axis=-1)<(frac*L)]
            pred_d,pred_o,pred_t,pred_p = sess.run([pd,po,pt,pp], feed_dict={msa:sub_a3m,ins:sub_ins,tape:tape_,idx:idx_})
        else:
            pred_d = np.zeros((L,L,37))
            pred_o = np.zeros((L,L,37))
            pred_t = np.zeros((L,L,37))
            pred_p = np.zeros((L,L,19))

        if args.windowed:

            # predict using discontinuous sliding windows
            k=0
            shift=8
            for window in [64,96,128]:

                # stop if window size is larger
                # than the size of the protein
                if 2*window > L:
                    break

                grids = np.arange(0,L-window,shift)
                wndws  = np.full(grids.shape[0], window, dtype=np.int)
                wndws[-1] = L - grids[-1]
                ngrids = grids.shape[0]
                print("grids:     ", grids)
                print("windows:   ", wndws)

                for i in range(ngrids):
                    for j in range(i+window//shift,ngrids):
                        sel = np.zeros((L)).astype(np.bool)
                        sel[grids[i]:grids[i]+wndws[i]] = True
                        sel[grids[j]:grids[j]+wndws[j]] = True

                        sub_a3m = a3m_[:,sel]
                        sub_ins = ins_[:,sel]
                        mask = np.sum(sub_a3m==20,axis=-1)<(frac*window)
                        sub_a3m = sub_a3m[mask]
                        sub_ins = sub_ins[mask]
                        sub_idx = idx_[sel]
                        sub_tape = tape_[:,sel,:]

                        mask2d = get_mask2d(wndws[i],wndws[j])

                        print("window%04d: [%d:%d]+[%d:%d] N(seq)=%d"%(k, grids[i], grids[i]+wndws[i], 
                                grids[j], grids[j]+wndws[j], np.sum(mask)))
                        out = sess.run([pd,po,pt,pp], feed_dict={msa:sub_a3m,ins:sub_ins,tape:sub_tape,idx:sub_idx})
                        sub_idx_2d = np.ix_(sub_idx,sub_idx.T)
                        counts[sub_idx_2d] += mask2d
                        pred_d[sub_idx_2d] += out[0]*mask2d[:,:,None]
                        pred_o[sub_idx_2d] += out[1]*mask2d[:,:,None]
                        pred_t[sub_idx_2d] += out[2]*mask2d[:,:,None]
                        pred_p[sub_idx_2d] += out[3]*mask2d[:,:,None]

                        k += 1


pred_d = pred_d / counts[:,:,None]
pred_o = pred_o / counts[:,:,None]
pred_t = pred_t / counts[:,:,None]
pred_p = pred_p / counts[:,:,None]

if args.roll:
    pred_d = np.roll(pred_d, 1, axis=-1)
    pred_o = np.roll(pred_o, 1, axis=-1)
    pred_t = np.roll(pred_t, 1, axis=-1)
    pred_p = np.roll(pred_p, 1, axis=-1)

# save
np.savez_compressed(args.NPZ,
    dist=pred_d,
    omega=pred_o,
    theta=pred_t,
    phi=pred_p)
