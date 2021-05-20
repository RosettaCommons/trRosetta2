import warnings, logging, os, sys
warnings.filterwarnings('ignore',category=FutureWarning)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
#import string
import sys,os
import json
from collections import namedtuple

#import time
from parsers import *
from coords6d import get_coords6d
from arguments import get_args
from utils import * #load_weights_msa,resblock
from ffindex import read_index,read_data,get_entry_by_name,read_entry_lines

FFindexDB = namedtuple("FFindexDB", "index, data")

config = tf.ConfigProto(
    gpu_options = tf.GPUOptions(allow_growth=True)
)

def main():

    ########################################################
    # 0. process argumets
    ########################################################
    wdir = os.path.abspath(os.path.dirname(__file__))
    args = get_args(wdir)

    # load network weights
    if args.hhr is not None:
        weights,params = load_weights(args.mdir+'/trrosetta_tbm/'+args.crop)
    else:
        weights,params = load_weights(args.mdir+'/trrosetta_msa/'+args.crop)


    ########################################################
    # 1. load inputs
    ########################################################

    # load MSA
    inputs = parse_a3m(args.msa)
    N,L = inputs['msa'].shape

    # load TAPE embeddings
    if args.tape is not None:
        #tape = np.load(args.tape)[:,1:-1,:]
        inputs.update({'tape':np.load(args.tape)[:,1:-1,:]})
    else:
        #tape = np.zeros((1,L,768),dtype=np.float32)
        inputs.update({'tape':np.zeros((1,L,768),dtype=np.float32)})
    
    # load templates
    if args.hhr is not None:
        ffdb = FFindexDB(read_index(args.ffdb+'_pdb.ffindex'),
                         read_data(args.ffdb+'_pdb.ffdata'))
        hhr = parse_hhr(args.hhr, ffdb.index, 999.)
        dbins = np.linspace(2.5, 20.0, 36)

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
            d,o,t,p = get_coords6d(xyz_tmp, 20.0)
            db = np.digitize(d,dbins)

            # residue indices in the query and the template
            sel = np.intersect1d(hit[1][:,1], idx_tmp, return_indices=True)[1]
            qmap = hit[1][sel,0]
            tmap = hit[1][sel,1]

            # format 1d features
            f1d_k = np.zeros((L,2), dtype=np.float32)
            f1d_k[qmap] = hit[1][sel,2:]
            f1d.append(f1d_k)

            # format 2d features
            f2d_k = np.zeros((L,L,7), dtype=np.float32)
            ij = np.ix_(qmap,qmap.T)
            f2d_k[:,:,0][ij] = db[tmap][:,tmap]
            f2d_k[:,:,1][ij] = o[tmap][:,tmap]
            f2d_k[:,:,2][ij] = t[tmap][:,tmap]
            f2d_k[:,:,3][ij] = p[tmap][:,tmap]
            f2d_k[:,:,4:][ij] = np.array(hit[2:])
            f2d.append(f2d_k)

        inputs.update({'t1d':np.array(f1d), 't2d':np.array(f2d)})
    

    ########################################################
    # 2. define the network
    ########################################################

    # placeholders shared by msa- and tbm- networks
    msa = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None,None))
    ins = tf.compat.v1.placeholder(dtype=tf.uint8, shape=(None,None))
    tape = tf.compat.v1.placeholder(dtype=tf.float32, shape=(1,None,768))
    idx = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None))

    if args.hhr is not None:
        # placeholders specific to the tbm-network
        t1d = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,None,2))
        t2d = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,None,None,7))
        pd,po,pt,pp = build_tbm_net(msa,ins,tape,idx,t1d,t2d, weights,params)
        keys = [msa,ins,tape,idx,t1d,t2d]
        vals = ['msa','ins','tape','idx','t1d','t2d']
    else:
        pd,po,pt,pp = build_msa_net(msa,ins,tape,idx, weights,params)
        keys = [msa,ins,tape,idx]
        vals = ['msa','ins','tape','idx']


    ########################################################
    # 3. inference
    ########################################################

    sess = tf.Session(config=config)
    
    # continuous cropping
    if args.crop == 'cont':
        
        # predict using full length seqeunce
        counts = np.full((L,L),1.0)
        fd = get_cont_crop(inputs,0,L,args.cov,args.maxseq)
        out = sess.run([pd,po,pt,pp], feed_dict={k:fd[v] for k,v in zip(keys,vals)})

        # predict using sliding windows
        #'''
        for window in [64,128,256]:

            # stop if window size is larger  than the size of the protein
            if window >= L: break

            # loop over all diagonal slices
            for i in range(L-window+1):
                fd_i = get_cont_crop(inputs,i,window,args.cov,args.maxseq)
                out_i = sess.run([pd,po,pt,pp], feed_dict={k:fd_i[v] for k,v in zip(keys,vals)})
                
                # merge with previous predictions
                counts[i:i+window,i:i+window] += 1
                for p,p_i in zip(out,out_i):
                    p[i:i+window,i:i+window] += p_i

    # discontinuous cropping
    else:
        
        counts = np.zeros((L,L))

        if L<=128:
            # predict using full alignment
            counts += 1
            fd = get_cont_crop(inputs,0,L,args.cov,args.maxseq)
            out = sess.run([pd,po,pt,pp], feed_dict={k:fd[v] for k,v in zip(keys,vals)})
        else:
            out = [np.zeros((L,L,nb)) for nb in (37,37,37,19)]

        # predict using discontinuous sliding windows
        shift=8
        for window in [64,96,128]:

            # stop if window size is larger than the size of the protein
            if 2*window > L: break

            grids = np.arange(0,L-window,shift)
            wndws  = np.full(grids.shape[0], window, dtype=np.int)
            wndws[-1] = L - grids[-1]
            ngrids = grids.shape[0]
            print("grids:   ", grids)
            print("windows: ", wndws)

            for i in range(ngrids):
                for j in range(i+window//shift,ngrids):

                    fd_i,idx_i = get_discont_crop(inputs,grids[i],grids[j],wndws[i],wndws[j],args.cov,args.maxseq)
                    out_i = sess.run([pd,po,pt,pp], feed_dict={k:fd_i[v] for k,v in zip(keys,vals)})

                    mask2d = get_mask2d(wndws[i],wndws[j])
                    sub_idx_2d = np.ix_(idx_i,idx_i)
                    counts[sub_idx_2d] += mask2d
                    for p,p_i in zip(out,out_i):
                        p[sub_idx_2d] += p_i*mask2d[:,:,None]

        
    ########################################################
    # 4. save results
    ########################################################

    np.savez_compressed(
        args.npz, 
        dist = np.roll(out[0]/counts[:,:,None], 1, axis=-1),
        omega = np.roll(out[1]/counts[:,:,None], 1, axis=-1), 
        theta = np.roll(out[2]/counts[:,:,None], 1, axis=-1), 
        phi = np.roll(out[3]/counts[:,:,None], 1, axis=-1)
    )


if __name__ == '__main__':
    main()

