# *trRosetta2* 
This package contains deep learning models and related scripts used by Baker group in CASP14.

## Installation

### Linux / Mac
```
# 1) clone package
git clone https://github.com/RosettaCommons/trRosetta2
cd trRosetta2

# 2) create conda environment
conda env create -f casp14-baker.yml
conda activate casp14-baker

# 3) download network weights [1.1G]
wget https://files.ipd.uw.edu/pub/trRosetta2/weights.tar.bz2
tar xf weights.tar.bz2

# 4) download and install third-party software
./install_dependencies.sh

# 5) download sequence and structure databases

# uniclust30 [46G]
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
tar xf UniRef30_2020_06_hhsuite.tar.gz

# structure templates [8.3G]
wget https://files.ipd.uw.edu/pub/trRosetta2/pdb100_2020Mar11.tar.gz
tar xf pdb100_2020Mar11.tar.gz
```

Obtain a [PyRosetta licence](https://els2.comotion.uw.edu/product/pyrosetta) and install the package in `casp14-baker` conda environment ([link](http://www.pyrosetta.org/dow)).


## Usage

```
mkdir -p examples/T1078
./run_pipeline example/T1078.fa example/T1078
```



## Links

* [Robetta server](https://robetta.bakerlab.org/)


## References

[1] I Anishchenko, M Baek, H Park, J Dauparas, N Hiranuma, S Mansoor, I Humphrey, D Baker. Protein structure prediction guided by predicted inter-residue geometries. 
In: [CASP14 Abstract Book, 2020](https://predictioncenter.org/casp14/doc/CASP14_Abstracts.pdf)

[2] H Park, M Baek, N Hiranuma, I Anishchenko, S Mansoor, J Dauparas, D Baker. Model refinement guided by an interplay between Deep-learning and Rosetta.
In: [CASP14 Abstract Book, 2020](https://predictioncenter.org/casp14/doc/CASP14_Abstracts.pdf)

[3] M Baek, I Anishchenko, H Park, I Humphrey, D Baker. Protein oligomer structure predictions guided by predicted inter-chain contacts.
In: [CASP14 Abstract Book, 2020](https://predictioncenter.org/casp14/doc/CASP14_Abstracts.pdf)
