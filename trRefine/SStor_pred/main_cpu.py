#!/usr/bin/env python

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
    opt.add_argument('-model_dir', '--model_dir', dest='model_dir', metavar='model_dir', default='%s/models/wo_templ'%SCRIPT_PATH, \
                         help='directory for checkpoint')
    #
    # input files for prediction
    opt.add_argument('-outprefix', dest='outprefix', default='SS', \
                     help="prefix of output npz file")
    opt.add_argument('-a3m_fn', '--a3m_fn', dest='a3m_fn', metavar='a3m_fn', default=None,\
                         help='input a3m file for prediction')
    opt.add_argument('-pdb_fn', '--pdb_fn', dest='pdb_fn', metavar='pdb_fn', default=None, \
                         help='input pdb file for prediction')
    opt.add_argument('-templ_fn', '--templ_fn', dest='templ_fn',  default=None, \
                         help='template information file in npz format')
    opt.add_argument('-db', dest='TMPDB', default='%s/../../pdb100_2020Mar11/pdb100_2020Mar11'%SCRIPT_PATH)
    #
    
    if len(sys.argv) == 1:
        opt.print_help()
        return
    #
    FLAGS = opt.parse_args()
    #
    use_cpu = True
    
    run_config = tf.ConfigProto()

    with tf.Session(config=run_config) as sess:
        if FLAGS.templ_fn == None: # w/o templates info
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
                        use_cpu=use_cpu,
                        use_templ=False)
        else: # w/ templates info
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
                        use_cpu=use_cpu,
                        use_templ=True)

        ML_model.predict(FLAGS)

if __name__ == '__main__':
    main()

