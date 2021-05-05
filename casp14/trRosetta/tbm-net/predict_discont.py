############################################################
# Changes:
# -------
# May 5: max template seqID cutoff changed 50 --> 999
############################################################

import tensorflow as tf
import numpy as np
import string
import sys,os
import json
import time
from features import get_features
from parsers import parse_a3m,parse_hhr,parse_pdb_lines
from arguments import get_args
from ffindex import read_index,read_data,get_entry_by_name,read_entry_lines
from coords6d import get_coords6d
from utils import *

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


def main():

    ########################################################
    # 0. process argumets
    ########################################################
    args = get_args()
    with open(args.MDIR + '/params.json') as jsonfile:
        params = json.load(jsonfile)


    ########################################################
    # 1. load inputs
    ########################################################

    # load FFindexDB with templates
    ffdb = FFindexDB(read_index(args.TMPDB+'_pdb.ffindex'),
                     read_data(args.TMPDB+'_pdb.ffdata'))

    # load MSA
    a3m_,ins_ = parse_a3m(args.MSA)
    a3m_ = a3m_[:20000]
    ins_ = ins_[:20000]
    L0 = a3m_.shape[1]

    # load TAPE features
    tape_ = np.load(args.TAPE)[:,1:-1,:]

    # load HHR
    hhr = parse_hhr(args.HHR, ffdb.index, 999.)

    # distance bins
    dstep = (params['DMAX'] - params['DMIN']) / params['DBINS']
    dbins = np.linspace(params['DMIN']+dstep, params['DMAX'], params['DBINS'])

    # process top templates
    f1d,f2d = [],[]
    for i,hit in enumerate(hhr[:args.ntmp]):

        # extract template from FFindexDB
        entry = get_entry_by_name(hit[0], ffdb.index)
        data = read_entry_lines(entry, ffdb.data)
        xyz_tmp,idx_tmp = parse_pdb_lines(data)
        idx_tmp = idx_tmp-1 # change residue indices to start from 0
        print("template%03d: %s len=%d prob=%.1f"%(i,hit[0],xyz_tmp.shape[1],hit[2]*100))

        # 6d coordinates
        d,o,t,p = get_coords6d(xyz_tmp, params['DMAX'])
        db = np.digitize(d,dbins)

        # residue indices in the query and the template
        sel = np.intersect1d(hit[1][:,1], idx_tmp, return_indices=True)[1]
        qmap = hit[1][sel,0]
        tmap = hit[1][sel,1]

        # format 1d features
        f1d_k = np.zeros((L0,2), dtype=np.float32)
        f1d_k[qmap] = hit[1][sel,2:]
        f1d.append(f1d_k)

        # format 2d features
        f2d_k = np.zeros((L0,L0,7), dtype=np.float32)
        ij = np.ix_(qmap,qmap.T)
        f2d_k[:,:,0][ij] = db[tmap][:,tmap]
        f2d_k[:,:,1][ij] = o[tmap][:,tmap]
        f2d_k[:,:,2][ij] = t[tmap][:,tmap]
        f2d_k[:,:,3][ij] = p[tmap][:,tmap]
        f2d_k[:,:,4:][ij] = np.array(hit[2:])#s0d[i][None,None,:]
        f2d.append(f2d_k)

    f1d = np.array(f1d)
    f2d = np.array(f2d)
    print("MSA+TAPE size:  %.2fmb"%((a3m_.nbytes+ins_.nbytes+tape_.nbytes)/1024**2))
    print("Templates size: %.2fmb"%((f1d.nbytes+f2d.nbytes)/1024**2))


    ########################################################
    # 2. load network weights
    ########################################################
    w,b,beta,gamma = load_weights_no1d(args, params)


    ########################################################
    # 3. define the network
    ########################################################

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()

    # network inputs
    with tf.name_scope('inputs'):
        msa = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None,None))
        ins = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None,None))
        tape = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1,None,768))
        idx = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None))
        t1d = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,None,2))
        t2d = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,None,None,7))

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

        # project down TAPE features
        #tape_reduced = Activation(Conv1d(tape, w1d[i], b1d[i]))

        # join MSA and TAPE features
        #inputs = tf.concat([msa_features,
        #                    tf.tile(tape_reduced[:,:,None,:], [1,1,ncol,1]),
        #                    tf.tile(tape_reduced[:,None,:,:], [1,ncol,1,1])], axis=-1)
        inputs = msa_features

        # process template features
        sgnl = Activation(Conv2d(t1d2d,w[i][0],b[i][0]))
        prob = tf.nn.softmax(Conv2d(t2d,w[i][1],b[i][1]), axis=0)
        inputs_tmp = tf.reduce_sum(sgnl*prob,axis=0)

        # merge all processed features together
        inputs = tf.concat([inputs_tmp[None,:,:,:],
                            msa_features], axis=-1)
                            #tf.tile(tape_reduced[:,:,None,:], [1,1,ncol,1]),
                            #tf.tile(tape_reduced[:,None,:,:], [1,ncol,1,1])], axis=-1)

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


    ########################################################
    # 4. inference
    ########################################################
    frac = 0.5
    with tf.compat.v1.Session(config=config) as sess:

        counts = np.zeros((L0,L0))

        idx_ = np.arange(L0)

        # predict using full alignment
        if L0<=128:
            counts += 1
            sub_a3m = a3m_[np.sum(a3m_==20,axis=-1)<(frac*L0)]
            sub_ins = ins_[np.sum(a3m_==20,axis=-1)<(frac*L0)]
            pred_d,pred_o,pred_t,pred_p = sess.run([pd,po,pt,pp], feed_dict={
                msa  : sub_a3m,
                ins  : sub_ins,
                tape : tape_,
                idx  : idx_,
                t1d  : f1d,
                t2d  : f2d })

        else:
            pred_d = np.zeros((L0,L0,37))
            pred_o = np.zeros((L0,L0,37))
            pred_t = np.zeros((L0,L0,37))
            pred_p = np.zeros((L0,L0,19))

        if args.windowed:

            # predict using discontinuous sliding windows
            k=0
            shift=8
            for window in [64,96,128]:

                # stop if window size is larger
                # than the size of the protein
                if 2*window > L0:
                    break

                grids = np.arange(0,L0-window,shift)
                wndws  = np.full(grids.shape[0], window, dtype=np.int)
                wndws[-1] = L0 - grids[-1]
                ngrids = grids.shape[0]
                print("grids:     ", grids)
                print("windows:   ", wndws)

                for i in range(ngrids):
                    for j in range(i+window//shift,ngrids):
                        sel = np.zeros((L0)).astype(np.bool)
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
                        out = sess.run([pd,po,pt,pp], feed_dict={
                            msa  : sub_a3m,
                            ins  : sub_ins,
                            tape : sub_tape,
                            idx  : sub_idx,
                            t1d  : f1d[:,sel],
                            t2d  : f2d[:,sel][:,:,sel] })

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


if __name__ == '__main__':
    main()

