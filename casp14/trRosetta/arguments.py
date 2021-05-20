import argparse

def get_args(wdir):

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('-i','--ia3m', type=str, required=True, dest='msa', help='multiple sequence alignment in A3M format (input)')
    parser.add_argument('-o','--onpz', type=str, required=True, dest='npz', help='network predictions (output)')
    parser.add_argument('-m','--mdir', type=str, required=True, dest='mdir', help='network predictions (output)')

    parser.add_argument('--tape', type=str, dest='tape', help='TAPE embeddings')
    parser.add_argument('--hhr', type=str, dest='hhr', help='template hits from HHsearch')
    parser.add_argument('--ffdb', type=str, dest='ffdb', default=wdir+'/../pdb100_2020Mar11/pdb100_2020Mar11', 
                        help='path to FFINDEX database of templates')
    parser.add_argument('--crop', type=str, dest='crop', default='cont', choices=['cont','discont'],
                        help='choose between continuous and discontinuous cropping modes')

    parser.add_argument('--ntmp', type=int, default=25, dest='ntmp', help='number of top HHsearch hits to use')
    parser.add_argument('--cov', type=float, default=0.5, dest='cov', help='clean MSA using this coverage cutoff')
    parser.add_argument('--maxseq', type=int, default=20000, dest='maxseq', help='max number of sequences in the cleaned MSA')

    args = parser.parse_args()

    return args

