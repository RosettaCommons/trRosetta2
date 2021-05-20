#!/usr/bin/env python
import os
import sys
import glob
import multiprocessing as mp
import numpy as np
from sklearn.cluster import AgglomerativeClustering

script_dir = os.path.dirname(__file__)
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='''TrRefine: Refinement of trRosetta outputs''')

    # Input npz file (outputs of msa-net, tbm-net)
    parser.add_argument("-msa_npz", required=True, \
                         help="output npz file of trRosetta msa-net")
    parser.add_argument("-tbm_npz", required=True, \
                         help="output npz file of trRosetta tbm-net")
    parser.add_argument("-pdb_dir_s", required=True, nargs='+', \
                         help="path to predicted pdb files by trRosetta")
    parser.add_argument("-a3m_fn", required=True, \
                         help="MSA file in a3m format")
    parser.add_argument("-hhr_fn", required=True, \
                         help="MSA file in hhr format")
    parser.add_argument('-n_core', required=True, type=int,\
                         help='The number of cores can ben used')
    parser.add_argument("--rescore", action='store_true', default=False)
    parser.add_argument("--use_cpu", action='store_true', default=False)
    args = parser.parse_args()

    return args

def rescore(pose):
    score_fxn = pyrosetta.create_score_function('ref2015_cart')
    score = score_fxn.score(pose)
    return score

def calc_lddt_dist(args):
    i, j, pose_s = args
    pose_i = pose_s[i]
    pose_j = pose_s[j]
    #
    lddt_1 = float(os.popen("/home/minkbaek/bin/lddt -c %s %s | grep Glob"%(pose_i, pose_j)).readlines()[-1].split()[-1])
    lddt_2 = float(os.popen("/home/minkbaek/bin/lddt -c %s %s | grep Glob"%(pose_j, pose_i)).readlines()[-1].split()[-1])
    lddt = (lddt_1 + lddt_2) / 2.0
    return 1 - lddt

def pick_rep(msa_npz, tbm_npz, pdb_dir_s, n_core, n_clust=10, rescore=False, cutoff=0.9, n_lowE=10):
    # pick 10 lowest E conformation from each method
    pdb_fn_s = list()
    score_s = list()
    pose_s = list()
    disto_s = list()
    for pdb_dir in pdb_dir_s:
        if 'pdb-msa' in pdb_dir:
            disto_s.extend([os.path.abspath(msa_npz) for i in range(n_lowE)])
        else:
            disto_s.extend([os.path.abspath(tbm_npz) for i in range(n_lowE)])
        fn_s = glob.glob("%s/model*.pdb"%(pdb_dir))
        fn_s.sort()
        #
        if not rescore:
            scores = list()
            for fn in fn_s:
                sc = float(os.popen("grep ^pose %s"%fn).readlines()[-1].split()[-1])
                scores.append(sc)
        else:
            import pyrosetta
            pyrosetta.init('-mute all')

            poses = list()
            for pdb_fn in fn_s:
                pose = pyrosetta.pose_from_file(pdb_fn)
                poses.append(pose.clone())

            # Setup multiprocessor
            print ("setup multiprocessor")
            n_core_pool = min(n_core, len(fn_s))
            pool = mp.Pool(n_core_pool)
            #
            # rescore all the inputs using ref2015_cart & filter out highE conf
            print ("rescoring")
            scores = pool.map(rescore, poses)
            pool.close()
            pool.join()
        scores = np.array(scores)
        idx = np.argsort(scores)
        fn_s = np.array(fn_s)
        score_s.append(scores[idx[:n_lowE]])
        sys.stdout.write("INFO: Pick 10 lowest E conformations from %s, Emin=%.3f / Emax=%.3f\n"%(pdb_dir, score_s[-1][0], score_s[-1][-1]))
        for i_fn in idx[:n_lowE]:
            pdb_fn_s.append(os.path.abspath(fn_s[i_fn]))

    #
    if not os.path.exists("rep_s"):
        os.mkdir("rep_s")
    os.chdir('rep_s')
    
    inputs = list()
    for i_pdb, pdb_fn in enumerate(pdb_fn_s):
        os.system("ln -sf %s candidate_%02d.pdb"%(pdb_fn, i_pdb))
        inputs.append("candidate_%02d.pdb"%i_pdb)
    with open("distogram_dan.list", 'wt') as fp:
        fp.write("\n".join(disto_s))
        fp.write("\n")
    with open("inpdb_dan.list", 'wt') as fp:
        fp.write("\n".join(inputs))
        fp.write("\n")
    #
    # run DAN-msa
    acc_fn_s = glob.glob("candidate_??_acc.npz")
    if len(acc_fn_s) < n_lowE*2:
        print ("run DAN_msa")
        os.system("python -u -W ignore %s/DAN-msa/ErrorPredictorMSA.py -p %d distogram_dan.list inpdb_dan.list ."%(script_dir, n_core))
    
    #
    tmp_score_s = list()
    tmp_fn_s = list()
    for i_pdb in range(n_lowE*2): 
        lddt = np.mean(np.load("candidate_%02d_acc.npz"%i_pdb)['lddt'])
        tmp_score_s.append(-1.0*lddt)
        tmp_fn_s.append("candidate_%02d.pdb"%i_pdb)
    tmp_score_s = np.array(tmp_score_s)
    #
    # Add best MSA model & tbm model
    model_s = list()
    msa_idx = np.argmin(tmp_score_s[:n_lowE])
    model_s.append(tmp_fn_s[msa_idx])
    #
    tbm_idx = np.argmin(tmp_score_s[n_lowE:])
    model_s.append(tmp_fn_s[tbm_idx+n_lowE])
    pdb_fn_s = list()
    score_s = list()
    for i, pdb_fn in enumerate(tmp_fn_s):
        if pdb_fn in model_s:
            continue
        pdb_fn_s.append(pdb_fn)
        score_s.append(tmp_score_s[i])
    # 
    score_s = np.array(score_s)
    Emin = np.min(tmp_score_s)
    Ecut = cutoff * Emin
    idx_s = np.where(score_s < Ecut)[0]
    n_clust = min(len(idx_s)+2, n_clust)
    score_s = np.concatenate([score_s[idx_s], np.array([tmp_score_s[msa_idx], tmp_score_s[tbm_idx+n_lowE]])])
    model_s = np.concatenate([np.array(pdb_fn_s)[idx_s], np.array(model_s)])
    n_str = len(model_s)
    #
    sys.stdout.write("INFO: After filtering based on E, Emin=%.3f Ecut=%.3f, %d / %d\n"%(Emin, Ecut, n_str, len(tmp_fn_s)))
    del pdb_fn_s, tmp_fn_s, tmp_score_s

    args = list()
    for i in range(n_str-1):
        for j in range(i+1, n_str):
            args.append((i,j,model_s))

    n_core_pool = min(n_core, len(args))
    pool = mp.Pool(n_core_pool)
    raw_dist = pool.map(calc_lddt_dist, args)
    pool.close()
    pool.join()
    dist = np.zeros((n_str, n_str), dtype=np.float)
    idx = np.triu_indices(n_str, k=1)
    dist[idx] = raw_dist
    dist = dist + dist.T
    #
    cluster = AgglomerativeClustering(n_clusters=n_clust, affinity='precomputed', linkage='single').fit(dist)
    #
    unique_labels = np.unique(cluster.labels_)
    rep_s = list()
    for label in unique_labels:
        idx = np.where(cluster.labels_==label)[0]
        #
        Emin_idx = np.argmin(score_s[idx])
        model = model_s[idx][Emin_idx]
        rep_s.append(model)
    #
    for i_rep, rep in enumerate(rep_s):
        pref = rep[:-4]
        os.system("ln -sf %s rep_%d.pdb"%(rep, i_rep))
        os.system("ln -sf %s_acc.npz rep_%d_acc.npz"%(pref, i_rep))
    
    with open("DONE", 'wt') as fp:
        fp.write("\n".join(rep_s))
        fp.write("\n")
    os.chdir('..')
    return rep_s

def main():
    args = get_args()

    # Pick 10 representative structures from the given pdbs
    if not os.path.exists("rep_s/DONE"):
        rep_s = pick_rep(args.msa_npz, args.tbm_npz, args.pdb_dir_s, args.n_core, rescore=args.rescore)
    else:
        rep_s = [line.strip() for line in open("rep_s/DONE")]


    # run SStor
    SS_fn_s = glob.glob("rep_s/*_SS.npz")
    if len(SS_fn_s) < len(rep_s):
        if args.use_cpu:
            os.system("python -u -W ignore %s/SStor_pred/main_multi_cpu.py -a3m_fn %s -hhr_fn %s -pdb_fn_s rep_s/rep_?.pdb"%(script_dir, args.a3m_fn, args.hhr_fn))
        else:
            print("python -u -W ignore %s/SStor_pred/main_multi.py -a3m_fn %s -hhr_fn %s -pdb_fn_s rep_s/rep_?.pdb"%(script_dir, args.a3m_fn, args.hhr_fn))
            os.system("python -u -W ignore %s/SStor_pred/main_multi.py -a3m_fn %s -hhr_fn %s -pdb_fn_s rep_s/rep_?.pdb"%(script_dir, args.a3m_fn, args.hhr_fn))
    #
    tor_s = list()
    for i in range(len(rep_s)):
        tor_s.append(np.load("rep_s/rep_%d_SS.npz"%i)['tor'].astype(np.float32)+1e-9)
    tor_s = np.stack(tor_s, axis=0)
    tor_s = np.mean(tor_s, axis=0)
    np.savez_compressed("rep_s/BBtor.npz", phi=tor_s[:,:36], psi=tor_s[:,36:72], omega=tor_s[:,72:])
        
    
    # run trRefine
    if not os.path.exists("t000_.trRefine.npz"):
        os.system("python -u -W ignore %s/trRefine/main.py -a3m_fn %s -pdb_fn_s rep_s/rep_?.pdb -acc_fn_s rep_s/rep_?_acc.npz -SS_fn_s rep_s/rep_?_SS.npz -npz_fn_s %s %s -out_fn t000_.trRefine.npz"%(script_dir, args.a3m_fn, args.msa_npz, args.tbm_npz))

main()
