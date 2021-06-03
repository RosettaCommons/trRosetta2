#!/usr/bin/env python
import glob
import os
import sys
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from model import ResNet_model

script_dir = os.path.abspath(os.path.dirname(__file__))

SCRIPT_PATH = os.path.dirname(__file__)

def main():
    import argparse
    opt = argparse.ArgumentParser\
            (description='''AngPred: Backbone torsion angle prediction using ResNet''')
    #
    # model save
    opt.add_argument('-model_dir', '--model_dir', dest='model_dir', metavar='model_dir', default='%s/models/w_templ'%SCRIPT_PATH, \
                         help='directory for checkpoint')
    #
    # input files for prediction
    opt.add_argument('-a3m_fn', required=True)
    opt.add_argument('-hhr_fn', default=None)
    opt.add_argument('-pdb_fn_s', required=True, nargs='+')
    opt.add_argument('-db', dest='TMPDB', default='%s/../../pdb100_2020Mar11/pdb100_2020Mar11'%SCRIPT_PATH)
    #
    if len(sys.argv) == 1:
        opt.print_help()
        return
    #
    FLAGS = opt.parse_args()
    #
    run_config = tf.ConfigProto()
    #
    a3m_s = list()
    pdb_s = list()
    hhr_s = list()
    out_s = list()
    for pdb_fn in FLAGS.pdb_fn_s:
        prefix = pdb_fn.split('/')[-1][:-4]
        out_fn = "rep_s/%s_SS"%(prefix)
        #
        a3m_s.append(FLAGS.a3m_fn)
        pdb_s.append(pdb_fn)
        hhr_s.append(FLAGS.hhr_fn)
        out_s.append(out_fn)

    with tf.Session(config=run_config) as sess:
        if hhr_s[0] != None:
            FLAGS.model_dir="%s/models/w_templ"%SCRIPT_PATH
            ML_model = ResNet_model(sess,
                        n_hidden_rnn=32,
                        attention_size=50,
                        n_1d_layer=20,
                        kernel_size=3,
                        n_feat_1d=64,
                        n_bottle_1d=32,
                        dilation=[1,2,4,8],
                        p_dropout = 0.2,
                        l2_coeff = 0.001,
                        use_templ=True,
                        use_cpu=True)
        else:
            FLAGS.model_dir="%s/models/wo_templ"%SCRIPT_PATH
            ML_model = ResNet_model(sess,
                        n_hidden_rnn=32,
                        attention_size=50,
                        n_1d_layer=20,
                        kernel_size=3,
                        n_feat_1d=64,
                        n_bottle_1d=32,
                        dilation=[1,2,4,8],
                        p_dropout = 0.2,
                        l2_coeff = 0.001,
                        use_templ=False,
                        use_cpu=True)

        ML_model.predict_multi(FLAGS, a3m_s, pdb_s, hhr_s, out_s)

if __name__ == '__main__':
    main()

