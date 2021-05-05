import argparse

def get_args():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("MSA", type=str, help="input multiple sequence alignment in A3M/FASTA format")
    parser.add_argument("TAPE", type=str, help="TAPE features")
    parser.add_argument("HHR", type=str, help="HHsearch hits")
    parser.add_argument("NPZ", type=str, help="predicted distograms and anglegrams")
    parser.add_argument('-m', type=str, required=True, dest='MDIR', help='folder with the pre-trained network')
    parser.add_argument('-t', type=str, required=True, dest='TMPDB', help='FFINDEX DB with templates')
    parser.add_argument('-n', type=int, default=25, dest='ntmp', help='number of top HHsearch hits to use')

    parser.add_argument('--windowed', dest='windowed', action='store_true', help='perform windowed predictions (takes longer)')
    parser.add_argument('--one-shot', dest='windowed', action='store_false')
    parser.add_argument("--roll", dest="roll", action='store_true', default=False, help="write outputs after roll index")
    parser.set_defaults(windowed=True)

    args = parser.parse_args()

    return args
