import argparse

def get_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("MSA", type=str, help="input multiple sequence alignment in A3M/FASTA format")
    parser.add_argument("TAPE", type=str, help="TAPE features")
    parser.add_argument("NPZ", type=str, help="predicted distograms and anglegrams")
    parser.add_argument('-m', type=str, required=True, dest='MDIR', help='folder with the pre-trained network')

    parser.add_argument('--windowed', dest='windowed', action='store_true', help='perform windowed predictions (takes longer)')
    parser.add_argument('--one-shot', dest='windowed', action='store_false')
    parser.add_argument("--roll", dest="roll", action='store_true', default=False, help="write outputs after roll index")
    parser.set_defaults(windowed=True)

    args = parser.parse_args()

    return args


def get_args_cmplx():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("MSA", type=str, help="input multiple sequence alignment in A3M/FASTA format")
    parser.add_argument("TAPE", type=str, help="TAPE features")
    parser.add_argument("NPZ", type=str, help="predicted distograms and anglegrams")
    parser.add_argument('-m', type=str, required=True, dest='MDIR', help='folder with the pre-trained network')
    parser.add_argument('-l1', type=int, required=True, dest='l1', help='length of protein 1')

    parser.add_argument('--windowed', dest='windowed', action='store_true', help='perform windowed predictions (takes longer)')
    parser.add_argument('--one-shot', dest='windowed', action='store_false')
    parser.set_defaults(windowed=True)

    args = parser.parse_args()

    return args

