import numpy as np
import scipy
import scipy.spatial
import string
import re

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

    return {'msa':msa, 'ins':ins}


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


def parse_pdb_lines(lines):

    N  = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="N"])
    Ca = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"])
    C  = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="C"])

    xyz = np.stack([N,Ca,C], axis=0)

    # indices of residues observed in the structure
    idx = np.array([int(l[22:26]) for l in lines if l[:4]=="ATOM" and l[12:16].strip()=="CA"])

    return [xyz,idx]


# parse HHsearch output
def parse_hhr(filename, ffindex, idmax=105.0):

    # labels present in the database
    label_set = set([i.name for i in ffindex])

    out = []

    with open(filename, "r") as hhr:

        # read .hhr into a list of lines
        lines = [s.rstrip() for _,s in enumerate(hhr)]

        # read list of all hits
        start = lines.index("") + 2
        stop = lines[start:].index("") + start
        hits = []
        for line in lines[start:stop]:

            # ID of the hit
            #label = re.sub('_','',line[4:10].strip())
            label = line[4:10].strip()

            # position in the query where the alignment starts
            qstart = int(line[75:84].strip().split("-")[0])-1

            # position in the template where the alignment starts
            tstart = int(line[85:94].strip().split("-")[0])-1

            hits.append([label, qstart, tstart, int(line[69:75])])

        # get line numbers where each hit starts
        start = [i for i,l in enumerate(lines) if l and l[0]==">"] # and l[1:].strip() in label_set]

        # process hits
        for idx,i in enumerate(start):

            # skip if hit is too short
            if hits[idx][3] < 10:
                continue

            # skip if template is not in the database
            if hits[idx][0] not in label_set:
                continue

            # get hit statistics
            p,e,s,_,seqid,sim,_,neff = [float(s) for s in re.sub('[=%]', ' ', lines[i+1]).split()[1::2]]

            # skip too similar hits
            if seqid > idmax:
                continue

            query = np.array(list(lines[i+4].split()[3]), dtype='|S1')
            tmplt = np.array(list(lines[i+8].split()[3]), dtype='|S1')

            simlr = np.array(list(lines[i+6][22:]), dtype='|S1').view(np.uint8)
            abc = np.array(list(" =-.+|"), dtype='|S1').view(np.uint8)
            for k in range(abc.shape[0]):
                simlr[simlr == abc[k]] = k

            confd = np.array(list(lines[i+11][22:]), dtype='|S1').view(np.uint8)
            abc = np.array(list(" 0123456789"), dtype='|S1').view(np.uint8)
            for k in range(abc.shape[0]):
                confd[confd == abc[k]] = k

            qj = np.cumsum(query!=b'-') + hits[idx][1]
            tj = np.cumsum(tmplt!=b'-') + hits[idx][2]

            # matched positions
            matches = np.array([[q-1,t-1,s-1,c-1] for q,t,s,c in zip(qj,tj,simlr,confd) if s>0])

            # skip short hits
            ncol = matches.shape[0]
            if ncol<10:
                continue

            # save hit
            #out.update({hits[idx][0] : [matches,p/100,seqid/100,neff/10]})
            out.append([hits[idx][0],matches,p/100,seqid/100,neff/10])

    return out
