import warnings, logging, os, sys
warnings.filterwarnings('ignore',category=FutureWarning)
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
from os import listdir
from os.path import isfile, isdir, join
import numpy as np
import pandas as pd
import multiprocessing

def main():
    #####################
    # Parsing arguments
    #####################
    parser = argparse.ArgumentParser(description="Error predictor network with predicted distogram",
                                     epilog="v0.0.1")
    parser.add_argument("dist_list",
                        action="store",
                        help="list of predicted distogram (npz format, key for distogram should be 'dist')")
    parser.add_argument("input_list",
                        action="store",
                        help="list of input pdb structures")
    parser.add_argument("outfolder",
                        action="store", nargs=argparse.REMAINDER,
                        help="output folder name. If a pdb path is passed this needs to be a .npz file. Can also be empty. Default is current folder or pdbname.npz")
    parser.add_argument('-prefix', default=None, help='prefix for input pdbs')
    parser.add_argument("--process",
                        "-p", action="store",
                        type=int,
                        default=1,
                        help="# of cpus to use for featurization (Default: 1)")
    args = parser.parse_args()

    ################################
    # Checking file availabilities #
    ################################
    #made outfolder an optional positinal argument. So check manually it's lenght and unpack the string
    if len(args.outfolder)>1:
        print(f"Only one output folder can be specified, but got {args.outfolder}", file=sys.stderr)
        return -1

    if len(args.outfolder)==0:
        args.outfolder = ""
    else:
        args.outfolder = args.outfolder[0]

    script_dir = os.path.dirname(__file__)
    base = os.path.join(script_dir, "models/")
    modelpath = base + "smTr"
        
    if not isdir(modelpath+"_rep1"):
        print("Model checkpoint does not exist", file=sys.stderr)
        return -1
        
    ##############################
    # Importing larger libraries #
    ##############################
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    script_dir = os.path.dirname(__file__)
    sys.path.insert(0, script_dir)
    import pyErrorPred
        
    num_process = 1
    if args.process > 1:
        num_process = args.process
        
    #########################
    # Getting samples names #
    #########################
    inputs = [line.strip() for line in open(args.input_list)]
    tmp_samples = [xxx.split('/')[-1][:-4] for xxx in inputs]
    if args.dist_list.split('.')[-1] == "npz": # single distogram for all
        distogram_s = [args.dist_list for s in tmp_samples]
    else:
        distogram_s = [line.strip() for line in open(args.dist_list)]
    samples = list()
    dist_s = list()
    for i_s, s in enumerate(tmp_samples):
        if os.path.exists(join(args.outfolder, "%s.npz"%s)): continue
        samples.append(s)
        dist_s.append(distogram_s[i_s])
    #
    if len(samples) < 1:
        return
    tmpoutputs = [join(args.outfolder, s)+".features.npz" for s in samples]
    arguments = [(inputs[i], tmpoutputs[i], False) for i in range(len(inputs)) if not isfile(tmpoutputs[i])]
    already_processed = [(inputs[i], tmpoutputs[i], False) for i in range(len(inputs)) if isfile(tmpoutputs[i])]
    
    if num_process == 1:
        for a in arguments:
            pyErrorPred.process(a)
    else:
        pool = multiprocessing.Pool(num_process)
        out = pool.map(pyErrorPred.process, arguments)

    ###########################
    # Prediction happens here #
    ###########################
    samples = [s for s in samples if isfile(join(args.outfolder, s+".features.npz"))]
    pyErrorPred.predict(samples,
                        dist_s,
                        modelpath,
                        args.outfolder)

    pyErrorPred.clean(samples,
                      args.outfolder)

if __name__== "__main__":
    main()
