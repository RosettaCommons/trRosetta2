#!/usr/bin/env python3

import os
import glob
import numpy as np
import scipy
from ffindex import *
from parsers import parse_hhr, parse_pdb_lines
from pyrosetta import *

init("-mute all")

MAXSEQ=20000

#============================================
# Global variables such as data directory path
#============================================
LABEL_dir = "/net/scratch/minkbaek/SStor/labels"
TEMPL_dir = "/net/scratch/minkbaek/trRefine/added/templ/sub_0/templates"
#TEMPL_dir = "/net/scratch/minkbaek/SStor/templs"
DECOY_DIR = "/projects/casp/dldata"


blosum_mtx = np.array([[  4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0],   
                       [ -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3],   
                       [ -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3],   
                       [ -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3],   
                       [  0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],   
                       [ -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2],   
                       [ -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2],   
                       [  0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3],   
                       [ -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3],   
                       [ -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3],   
                       [ -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1],   
                       [ -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2],   
                       [ -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1],   
                       [ -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1],   
                       [ -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2],   
                       [  1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2],   
                       [  0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0],   
                       [ -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3],   
                       [ -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1],   
                       [  0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4]]).astype(np.float32)
blosum_mtx /= 10.0 # same as Nao's

tip_atoms = {"ALA": "CB", "CYS": "SG", "ASP": "CG", "GLU": "CD",
             "PHE": "CZ", "GLY": "CA", "HIS":"NE2", "ILE":"CD1",
             "LYS": "NZ", "LEU": "CG", "MET": "SD", "ASN": "CG", 
             "PRO": "CG", "GLN": "CD", "ARG": "CZ", "SER": "OG", 
             "THR":"OG1", "VAL": "CB", "TRP":"CH2", "TYR": "OH"}

def f(X, cutoff=3, scaling=3.0):
    X_prime = np.maximum(X, np.zeros_like(X) + cutoff) - cutoff
    return np.arcsinh(X_prime)/scaling

def blosum(seq):
    return blosum_mtx[seq]

def t2q(tmplt,nres):

    dmin = 2.0
    dmax = 20.0

    qmap = tmplt['qmap']
    L = qmap.shape[0]

    # bin distance matrix
    nbins = 18
    bins = np.linspace(dmin, dmax, nbins+1)
    dist = tmplt['dist6d']
    dist[dist<0.001] = 999.9
    dbin = np.digitize(dist, bins).astype(np.float32)
    dbin[dbin > nbins] = 0

    f1d = np.zeros((nres,4), dtype=np.float32)
    f1d[qmap] = np.concatenate([
        tmplt['phi'][:,None],
        tmplt['psi'][:,None],
        tmplt['scores'][:,None],
        tmplt['conf'][:,None]
    ], axis=-1)

    f2d = np.zeros((nres,nres,6), dtype=np.float32)
    f2d[np.ix_(qmap,qmap)] = np.concatenate([
        dbin[:,:,None],
        tmplt['omega6d'][:,:,None],
        tmplt['theta6d'][:,:,None],
        tmplt['phi6d'][:,:,None],
        np.tile(tmplt['stats'][None,None,:],[L,L,1])
    ], axis=-1)

    return [f1d,f2d]

def subsample_msa(msa):
    n_sub = int(msa.shape[0]/2)
    if n_sub < 5:
        return msa
    seq = msa[0]
    tmp = msa[1:]
    np.random.shuffle(tmp)
    return np.concatenate([seq[np.newaxis,:], tmp[:n_sub-1,:]], axis=0)

def get_seqsep(n_res):
    pos = np.arange(n_res)
    pos = pos.astype(np.float32)
    tiled_pos = np.tile(pos, (n_res, 1))
    seqsep = np.abs(tiled_pos - tiled_pos.T)/100.0 - 1.0
    return seqsep

def get_labels(pdb, noise_level=0.0):
    label_fn = os.path.join(LABEL_dir, pdb + ".npz")
    data = np.load(label_fn)
    #
    msa = subsample_msa(data['msa'])
    seq = blosum(msa[0])
    #
    dssp = data['dssp'].astype(np.float32)
    phi = data['phi'].astype(np.float32)
    psi = data['psi'].astype(np.float32)
    omg = data['omg'].astype(np.float32)
    omg = np.stack((1-omg, omg), axis=-1)

    return dssp, phi, psi, omg, msa, seq

def get_input_features(filename, is_train=True, mask_diag=8):
    fn_s = glob.glob("%s/%s/*.npz"%(DECOY_DIR, filename))
    np.random.shuffle(fn_s)
    fn = fn_s[0]
    #
    data = np.load(fn)
    #
    seqsep = get_seqsep(data['maps'].shape[0])
    #
    # structure-based features
    d_CB = f(data['maps'][:,:,0].astype(np.float32))
    o_map = np.stack([data["omega6d"], data["theta6d"], data["phi6d"]], axis=-1).astype(np.float32)
    o_map = np.concatenate([np.sin(o_map), np.cos(o_map)], axis=-1)
    feat_2d = np.concatenate((d_CB[:,:,None], o_map, seqsep[:,:,None]), axis=-1)
    if is_train: # add gaussian noise in training stage to avoid overfitting
        scale = np.random.uniform(low=0.05, high=0.15)
        noise = np.random.normal(scale=scale, size=feat_2d.shape).astype(np.float32)
        feat_2d += noise
    feat_2d = np.nan_to_num(feat_2d)
    #
    if mask_diag > -1:
        for i in range(feat_2d.shape[-1]):
            for k in range(mask_diag):
                np.fill_diagonal(feat_2d[k:,:,i], 0.0)
                np.fill_diagonal(feat_2d[:,k:,i], 0.0)
    #
    # Template information
    nres = data['maps'].shape[0]
    tlist = glob.glob("%s/%s/*.npz"%(TEMPL_dir, filename))
    np.random.shuffle(tlist)
    n_templ = np.random.randint(10, 26)
    tlist = tlist[:n_templ]
    pthreads = [t2q(np.load(f), nres) for f in tlist]
    pth_1d = np.stack([p[0] for p in pthreads], axis=0)
    pth_2d = np.stack([p[1] for p in pthreads], axis=0)
    #
    return feat_2d, pth_1d, pth_2d

def load_train_data(pdb, noise_level=0.0, tag=None, is_train=True, mask_diag=8):
    #
    SS_label, phi_label, psi_label, omg_label, msa, seq = get_labels(pdb, noise_level=noise_level)
    feat_2d, pth_1d, pth_2d = get_input_features(pdb, is_train=is_train, mask_diag=mask_diag)
    #
    
    return seq, msa, feat_2d, pth_1d, pth_2d, SS_label, phi_label, psi_label, omg_label

# TODO
# input feature generation for prediction
def read_msa(a3m_fn):
    import string

    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file
    with open(a3m_fn) as fp:
        for line in fp:
            if line[0] == '>': continue
            seqs.append(line.rstrip().translate(table)) # remove lowercase letters
    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    msa[msa > 20] = 20
    return msa

def get_dihedrals(a, b, c, d):
    b0 = -1.0*(b - a)
    b1 = c - b
    b2 = d - c

    b1 /= np.linalg.norm(b1, axis=-1)[:,None]

    v = b0 - np.sum(b0*b1, axis=-1)[:,None]*b1
    w = b2 - np.sum(b2*b1, axis=-1)[:,None]*b1

    x = np.sum(v*w, axis=-1)
    y = np.sum(np.cross(b1, v)*w, axis=-1)

    return np.arctan2(y, x)

def get_angles(a, b, c):
    v = a - b
    v /= np.linalg.norm(v, axis=-1)[:,None]
    
    w = c - b
    w /= np.linalg.norm(w, axis=-1)[:,None]
    
    x = np.sum(v*w, axis=1)

    return np.arccos(np.clip(x, -1.0, 1.0))

def get_dist_ori_maps(model, resNo_s, n_res, dmax=20.0, return_sep=False):
    atm_names = list()
    for i in range(1, model.size()+1):
        if model.residue(i).name()[:3] == "GLY":
            atm_names.append('CA')
        else:
            atm_names.append('CB')
    #
    CA  = np.zeros((n_res, 3), dtype=np.float32)
    N   = np.zeros((n_res, 3), dtype=np.float32)
    O   = np.zeros((n_res, 3), dtype=np.float32)
    C   = np.zeros((n_res, 3), dtype=np.float32)
    CB  = np.zeros((n_res, 3), dtype=np.float32)
    #
    for idx, resNo in enumerate(resNo_s):
        i = idx + 1 # resNo in pose
        CA[resNo, :] = np.array(model.residue(i).atom("CA").xyz())
        N[resNo, :] = np.array(model.residue(i).atom("N").xyz())
        O[resNo, :] = np.array(model.residue(i).atom("O").xyz())
        C[resNo, :] = np.array(model.residue(i).atom("C").xyz())
        CB[resNo, :] = np.array(model.residue(i).atom(atm_names[idx]).xyz())
    #
    mask = np.zeros((n_res, n_res), dtype=np.float32)
    for i in resNo_s:
        for j in resNo_s:
            mask[i,j] = 1.0
    #
    dist_CB_CB = scipy.spatial.distance.cdist(CB, CB, metric='euclidean')
    #
    dist_maps = f(dist_CB_CB[:,:,None])
    #
    # recreate Cb from given N, Ca, C
    b = CA - N
    c = C - CA
    a = np.cross(b, c)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + CA
    #
    # neighbor search
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)
    #
    # indices of contacting residues
    idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i] if (i != j) and (i in resNo_s) and (j in resNo_s)]).T
    idx0 = idx[0]
    idx1 = idx[1]
    #
    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.zeros((n_res, n_res))
    ang = get_dihedrals(CA[idx0], CB[idx0], CB[idx1], CA[idx1])
    omega6d[idx0, idx1] = ang
    #
    # matrix of polar coord theta
    theta6d = np.zeros((n_res, n_res))
    ang = get_dihedrals(N[idx0], CA[idx0], Cb[idx0], Cb[idx1])
    theta6d[idx0, idx1] = ang 
    #
    # matrix of polar coord phi
    phi6d = np.zeros((n_res, n_res))
    ang = get_angles(CA[idx0], Cb[idx0], Cb[idx1])
    phi6d[idx0, idx1] = ang
    #
    if return_sep:
        dist6d = np.zeros((nres, nres), dtype=np.float32)
        dist6d[idx0, idx1] = np.linalg.norm(Cb[idx1]-Cb[idx0], axis=-1)
        return dist6d, omega6d, theta6d, phi6d
    #
    ori_map = np.stack([omega6d, theta6d, phi6d], axis=-1).astype(np.float32)
    ori_map = np.concatenate([np.sin(ori_map), np.cos(ori_map)], axis=-1)
    #
    ori_map = ori_map * mask[:,:,None]
    dist_maps = dist_maps * mask[:,:,None]
    #
    return dist_maps, ori_map

def get_phipsi(xyz, tmap):
    nres = xyz.shape[1]
    
    N = xyz[0]
    Ca = xyz[1]
    C = xyz[2]

    C_prev = np.insert(C, 0, np.nan, axis=0)
    N_next = np.insert(N, nres, np.nan, axis=0)
    
    phi = get_dihedrals(C_prev[:-1,:], N, Ca, C)
    psi = get_dihedrals(N, Ca, C, N_next[1:,:])
    phi[0] = 0.0
    psi[-1] = 0.0
    tor = np.stack((phi, psi), axis=-1)[tmap]
    return tor

# get 6d coordinates from x,y,z coords of N,Ca,C atoms
def get_coords6d(xyz, dmax):

    nres = xyz.shape[1]

    # three anchor atoms
    N  = xyz[0]
    Ca = xyz[1]
    C  = xyz[2]

    # recreate Cb given N,Ca,C
    b = Ca - N
    c = C - Ca
    a = np.cross(b, c)
    Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + Ca

    # fast neighbors search to collect all
    # Cb-Cb pairs within dmax
    kdCb = scipy.spatial.cKDTree(Cb)
    indices = kdCb.query_ball_tree(kdCb, dmax)

    # indices of contacting residues
    idx = np.array([[i,j] for i in range(len(indices)) for j in indices[i] if i != j]).T
    idx0 = idx[0]
    idx1 = idx[1]

    # Cb-Cb distance matrix
    dist6d = np.zeros((nres, nres))
    dist6d[idx0,idx1] = np.linalg.norm(Cb[idx1]-Cb[idx0], axis=-1)

    # matrix of Ca-Cb-Cb-Ca dihedrals
    omega6d = np.zeros((nres, nres))
    omega6d[idx0,idx1] = get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])

    # matrix of polar coord theta
    theta6d = np.zeros((nres, nres))
    theta6d[idx0,idx1] = get_dihedrals(N[idx0], Ca[idx0], Cb[idx0], Cb[idx1])

    # matrix of polar coord phi
    phi6d = np.zeros((nres, nres))
    phi6d[idx0,idx1] = get_angles(Ca[idx0], Cb[idx0], Cb[idx1])

    return dist6d, omega6d, theta6d, phi6d

def get_2D_str_feat(model, n_res):
    resNo_s = list()
    for i in range(1, model.size()+1):
        resNo = model.pdb_info().number(i) # resNo in PDB file
        resNo_s.append(resNo-1) # -1 to make it as python array index
    resNo_s = np.array(resNo_s)
    d_map, o_map = get_dist_ori_maps(model, resNo_s, n_res)
    seqsep = get_seqsep(n_res)
    return np.concatenate((d_map, o_map, seqsep[:,:,None]), axis=-1) 

def make_input_features(a3m_fn, pdb_fn=None, hhr_fn=None, ffdb=None, mask_diag=8, seqID=999.0):
    if isinstance(a3m_fn, str):
        msa = read_msa(a3m_fn)
    else:
        msa = a3m_fn
    seq = blosum(msa[0])
    if len(msa) > MAXSEQ:
        msa = msa[:MAXSEQ]
    #
    if pdb_fn == None: return seq, msa
    model = pose_from_file(pdb_fn)
    feat_2d = get_2D_str_feat(model, len(msa[0]))
    for i in range(feat_2d.shape[-1]):
        for k in range(mask_diag):
            np.fill_diagonal(feat_2d[k:,:,i], 0.0)
            np.fill_diagonal(feat_2d[:,k:,i], 0.0)
  
    pth_1d = list()
    pth_2d = list()
    if hhr_fn != None:
        L0 = len(msa[0])
        dmin = 2.0
        dmax = 20.0
        nbins = 18
        dbins = np.linspace(dmin, dmax, nbins+1)
        hhr = parse_hhr(hhr_fn, ffdb.index, seqID)
        for i, hit in enumerate(hhr[:10]):
            # extract template from FFindexDB
            entry = get_entry_by_name(hit[0], ffdb.index)
            data = read_entry_lines(entry, ffdb.data)
            xyz_tmp, idx_tmp = parse_pdb_lines(data)
            idx_tmp = idx_tmp-1 # change residue indices to start from 0

            # 6d coordinates
            d, o, t, p = get_coords6d(xyz_tmp, 20.0)
            db = np.digitize(d,dbins)

            # residue indices in the query and the template
            sel = np.intersect1d(hit[1][:,1], idx_tmp, return_indices=True)[1]
            qmap = hit[1][sel,0]
            tmap = hit[1][sel,1]

            # format 1d features
            f1d_a = np.zeros((L0,2), dtype=np.float32)
            f1d_a[qmap] = get_phipsi(xyz_tmp, tmap)
            f1d_k = np.zeros((L0,2), dtype=np.float32)
            f1d_k[qmap] = hit[1][sel,2:]
            pth_1d.append(np.concatenate((f1d_a, f1d_k), axis=-1))

            # format 2d features
            f2d_k = np.zeros((L0,L0,6), dtype=np.float32)
            ij = np.ix_(qmap,qmap.T)
            f2d_k[:,:,0][ij] = db[tmap][:,tmap]
            f2d_k[:,:,1][ij] = o[tmap][:,tmap]
            f2d_k[:,:,2][ij] = t[tmap][:,tmap]
            f2d_k[:,:,3][ij] = p[tmap][:,tmap]
            f2d_k[:,:,4:][ij] = np.array(hit[2:])#s0d[i][None,None,:]
            pth_2d.append(f2d_k)

        pth_1d = np.array(pth_1d)
        pth_2d = np.array(pth_2d)
    #
    return seq, msa, feat_2d, pth_1d, pth_2d
