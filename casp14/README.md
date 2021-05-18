# *Scripts used in CASP14 by BAKER group*
Deep learning models and related scripts used by Baker group in CASP14.

## Installation

1. Clone the package
```
git clone https://github.com/RosettaCommons/deep_learning_public
cd deep_learning_public/casp14
```

2. Download network weights
```
wget https://files.ipd.uw.edu/pub/trRosetta2/weights.tar.bz2
tar xf weights.tar.bz2
```

3. Install python dependencies
```
conda env create -f casp14-gpu.yml
conda activate casp14-gpu
```

Obtain and install PyRosetta in the same conda environment (link to licensing).


4. Download and install third-party software
```
./install_dependencies.sh
```

5. Download sequence and structure databases
```
# uniclust30 for hh-suite
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
tar xf UniRef30_2020_06_hhsuite.tar.gz

# database of pdb templates
wget https://files.ipd.uw.edu/pub/trRosetta2/pdb100_2020Mar11.tar.bz2
tar xf pdb100_2020Mar11.tar.bz2
```
