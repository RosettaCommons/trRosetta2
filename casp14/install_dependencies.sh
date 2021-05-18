#!/bin/bash

# clone and install hh-suite
git clone https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
(
cmake -DCMAKE_INSTALL_PREFIX=. ..
make -j 4 && make install
) > install.stdout 2> install.stderr
cd ../..

# clone and install psipred
git clone https://github.com/psipred/psipred
cd psipred/src
(
make && make install
) > install.stdout 2> install.stderr
cd ../..

# download lddt
echo "downloading lddt . . ."
#wget https://openstructure.org/static/lddt-macosx.zip -O lddt.zip
wget https://openstructure.org/static/lddt-linux.zip -O lddt.zip
unzip -d lddt -j lddt.zip

# download legacy blast
echo "downloading blast . . ."
#wget https://ftp.ncbi.nlm.nih.gov/blast/executables/legacy.NOTSUPPORTED/2.2.26/blast-2.2.26-universal-macosx.tar.gz -O blast-2.2.26.tar.gz
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/legacy.NOTSUPPORTED/2.2.26/blast-2.2.26-x64-linux.tar.gz -O blast-2.2.26.tar.gz
tar xf blast-2.2.26.tar.gz
