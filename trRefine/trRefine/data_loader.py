#!/usr/bin/env python

import os
import glob
import numpy as np
import scipy
from pyrosetta import *

init("-mute all")

#============================================
# Global variables such as data directory path
#============================================
LABEL_DIR = "/net/scratch/minkbaek/trRefine/labels.v2"
DECOY_DIR = "/net/scratch/minkbaek/trRefine/features.v2"

eps = 1e-9

def subsample_msa(msa):
    n_sub = int(msa.shape[0]/2)
    if n_sub < 5:
        return msa
    seq = msa[0]
    tmp = msa[1:]
    np.random.shuffle(tmp)
    return np.concatenate([seq[np.newaxis,:], tmp[:n_sub-1,:]], axis=0)

def get_labels(pdb):
    label_fn = os.path.join(LABEL_DIR, pdb + ".npz")
    data = np.load(label_fn)
    #
    # distance matrix
    dbin = data['dist']
    #
    # bin omega
    obin = data['omega']
    tbin = data['theta']
    pbin = data['phi']

    return dbin, obin, tbin, pbin

def get_input_features(filename, tag=None):
    if tag != None: # use specific subsampled msa
        sub_dir = "%s/%s/sub_%d"%(DECOY_DIR, filename, tag)
        sub_idx = sub_dir.split('/')[-1]
    else:
        sub_dir = np.random.choice(glob.glob("%s/%s/sub_?"%(DECOY_DIR, filename)))
        sub_idx = sub_dir.split('/')[-1]
    #
    msa = np.load("%s/msa.npy"%sub_dir)
    nres = len(msa[0])
    #
    dat = np.load("%s/map.npz"%sub_dir)
    naive_6d = dat['naive'].astype(np.float32)
    templ_6d = dat['templ'].astype(np.float32)
    #
    naive_s = glob.glob("%s/naive???.npz"%sub_dir)
    templ_s = glob.glob("%s/tmpl???.npz"%sub_dir)
    #
    np.random.shuffle(naive_s)
    np.random.shuffle(templ_s)
    rest_s = list()
    rest_s.extend(naive_s[1:])
    rest_s.extend(templ_s[1:])
    np.random.shuffle(rest_s)
    #
    feat_s = list()
    feat_s.append(naive_s[0])
    feat_s.append(templ_s[0])
    feat_s.extend(rest_s)
    #
    if tag != None: # validation time
        n_decoy = 10 # always use 10 decoys
    else:
        if nres < 200:
            n_decoy = np.random.randint(low=5, high=11)
        else:
            n_decoy = np.random.randint(low=3, high=6)
    #
    decoy_s = list()
    SS = np.zeros((nres, 9), dtype=np.float32)
    for feat_fn in feat_s[:n_decoy]:
        feat = np.load(feat_fn)
        decoy_s.append(feat['f2d'].astype(np.float32))
        #SS += feat['f1d'].astype(np.float32)
    #SS = SS / float(n_decoy)

    decoy_s = np.stack(decoy_s, axis=0)
    #
    return msa, SS, np.stack((naive_6d, templ_6d), axis=0), decoy_s

def load_train_data(pdb_in, tag=None):
    pdb = pdb_in.decode('utf-8')
    #
    label_dist, label_omega, label_theta, label_phi = get_labels(pdb)
    msa, SS, maps, strs = get_input_features(pdb, tag=tag)
    #
    return msa, SS, maps, strs, label_dist, label_theta, label_phi, label_omega

valid_info = {}
def setup_valid_dataset(valid_fn):
    global valid_info
    with open(valid_fn) as fp:
        for line in fp:
            x = line.split()
            valid_info[x[0]] = int(x[1])

def load_valid_data(pdb_in, tag=None):
    pdb = pdb_in.decode('utf-8')
    #
    label_dist, label_omega, label_theta, label_phi = get_labels(pdb)
    msa, SS, maps, strs = get_input_features(pdb, tag=valid_info[pdb])
    #
    return msa, SS, maps, strs, label_dist, label_theta, label_phi, label_omega

BB_ATOMS = ['N', 'CA', 'C']
def get_coords_pdb(pdb_fn_s, n_res):
    atom_read = np.zeros((3, len(pdb_fn_s), n_res, 3), dtype=np.float32)
    #
    for i_pdb, pdb_fn in enumerate(pdb_fn_s):
        with open(pdb_fn) as fp:
            for line in fp:
                if not line.startswith("ATOM"): continue
                resNo = int(line[22:26])
                resName = line[17:20]
                atmName = line[12:16].strip()
                #
                if atmName in BB_ATOMS:
                    xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    atom_read[BB_ATOMS.index(atmName), i_pdb, resNo-1, :] = xyz
    #
    atom_s = list()
    for i in range(len(BB_ATOMS)):
        atom_s.append(atom_read[i])
    return atom_s 

def read_a3m(seq_fn):
    import string
    seqs = []
    #
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file
    with open(seq_fn) as fp:
        for line in fp:
            if line[0] == '>': continue
            seqs.append(line.rstrip().translate(table)) # remove lowercase letters
    #
    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    msa[msa > 20] = 20

    return msa.astype(np.uint8)

def process_inputs(a3m_fn, npz_fn_s, pdb_fn_s, acc_fn_s, SS_fn_s):
    msa = read_a3m(a3m_fn)
    n_res = len(msa[0])
    SS = np.zeros((n_res, 9), dtype=np.float32)
    for SS_fn in SS_fn_s:
        SS += np.load(SS_fn)['ss9'].astype(np.float32)
    SS = SS / float(len(SS_fn_s))
    #
    N, CA, C = get_coords_pdb(pdb_fn_s, n_res)
    #
    pos = np.arange(n_res)
    tiled_pos = np.tile(pos, (n_res,1))
    seqsep = np.abs(tiled_pos - tiled_pos.T) / 100.0 - 1.0
    seqsep = seqsep.astype(np.float16)
    #
    # predicted 6D crd distribution
    pred_6d = list()
    for npz_fn in npz_fn_s:
        dat = np.load(npz_fn)
        #
        item = list()
        for key in ['dist', 'omega', 'theta', 'phi']:
            item.append(dat[key].astype(np.float32))
        item.append(seqsep[:,:,None])
        pred_6d.append(np.concatenate(item, axis=-1))
    pred_6d = np.stack(pred_6d, axis=0)
    #
    esto = list()
    for acc_fn in acc_fn_s:
        esto.append(np.load(acc_fn)['estogram'].astype(np.float32))
    esto = np.stack(esto, axis=0)
    #
    return msa, SS, N, CA, C, esto, seqsep, pred_6d
