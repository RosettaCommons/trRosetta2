import numpy as np
import scipy
import scipy.spatial
import string

to1letter = {
    "ALA":'A', "ARG":'R', "ASN":'N', "ASP":'D', "CYS":'C',
    "GLN":'Q', "GLU":'E', "GLY":'G', "HIS":'H', "ILE":'I',
    "LEU":'L', "LYS":'K', "MET":'M', "PHE":'F', "PRO":'P',
    "SER":'S', "THR":'T', "TRP":'W', "TYR":'Y', "VAL":'V'}

# read A3M and convert letters into
# integers in the 0..20 range,
# also keep track of insertions
def parse_a3m(filename):

    msa = []
    ins = []

    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    for line in open(filename,"r"):

        # skip labels
        if line[0] == '>':
            continue

        # remove right whitespaces
        line = line.rstrip()

        # remove lowercase letters and append to MSA
        msa.append(line.translate(table))

        # sequence length
        L = len(msa[-1])

        # 0 - match or gap; 1 - insertion
        a = np.array([0 if c.isupper() or c=='-' else 1 for c in line])
        i = np.zeros((L))

        if np.sum(a) > 0:
            # positions of insertions
            pos = np.where(a==1)[0]

            # shift by occurrence
            a = pos - np.arange(pos.shape[0])

            # position of insertions in cleaned sequence
            # and their length
            pos,num = np.unique(a, return_counts=True)

            # append to the matrix of insetions
            i[pos] = num

        ins.append(i)

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in msa], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    ins = np.array(ins, dtype=np.uint8)

    return msa,ins


# randomly subsample the alignment
def subsample_msa(msa,ins):

    nr, nc = msa.shape

    # do not subsample shallow MSAs
    if nr < 10:
        return msa,ins

    # number of sequences to subsample
    n = int(10.0**np.random.uniform(np.log10(nr))-10.0)
    if n <= 0:
        return np.reshape(msa[0],[1,nc]),np.reshape(ins[0],[1,nc])

    # n unique indices
    sample = sorted(random.sample(range(1, nr-1), n))

    # stack first sequence with n randomly selected ones
    msa_new = np.vstack([msa[0][None,:], msa[1:][sample]])
    ins_new = np.vstack([ins[0][None,:], ins[1:][sample]])

    return msa_new.astype(np.uint8),ins_new.astype(np.uint8)


# read and extract xyz coords of N,Ca,C atoms
# from a PDB file
def parse_pdb(filename):

    lines = open(filename,'r').readlines()

    N  = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="N"])
    Ca = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"])
    C  = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="C"])

    xyz = np.stack([N,Ca,C], axis=0)

    # indices of residues observed in the structure
    idx = np.array([int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"])

    return xyz,idx
