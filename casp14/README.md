# *trRosetta2* 
This package contains deep learning models and related scripts used by Baker group in CASP14.

## Installation

### Linux / Mac
```
# 1) clone packcage
git clone https://github.com/RosettaCommons/deep_learning_public
cd deep_learning_public/casp14

# 2) download network weights
wget https://files.ipd.uw.edu/pub/trRosetta2/weights.tar.bz2
tar xf weights.tar.bz2

# 3) create conda environment
conda env create -f casp14-baker.yml
conda activate casp14-baker

# 4) download and install third-party software
./install_dependencies.sh

# 5) download sequence and structure databases

# uniclust30 for hh-suite
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
tar xf UniRef30_2020_06_hhsuite.tar.gz

# database of pdb templates
wget https://files.ipd.uw.edu/pub/trRosetta2/pdb100_2020Mar11.tar.gz
tar xf pdb100_2020Mar11.tar.gz
```

Obtain a [PyRosetta licence](https://els2.comotion.uw.edu/product/pyrosetta) and install the package in `casp14-baker` conda environment ([link](http://www.pyrosetta.org/dow)).


## Usage

```
mkdir -p examples/T1078
./run_pipeline example/T1078.fa example/T1078
```

