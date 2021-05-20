#!/bin/bash

SYS=`uname`
if [ "$SYS" = "Linux" ] || [ "$SYS" = "Darwin" ];
then
    echo "installing for Linux / Mac"
else
    echo "cannot detect OS type"
    exit 1
fi

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
if [ "$SYS" = "Darwin" ]
then
    wget https://openstructure.org/static/lddt-macosx.zip -O lddt.zip
else
    wget https://openstructure.org/static/lddt-linux.zip -O lddt.zip
fi
unzip -d lddt -j lddt.zip

# download legacy blast
echo "downloading blast . . ."
if [ "$SYS" = "Darwin" ]
then
    wget https://ftp.ncbi.nlm.nih.gov/blast/executables/legacy.NOTSUPPORTED/2.2.26/blast-2.2.26-universal-macosx.tar.gz -O blast-2.2.26.tar.gz
else
    wget https://ftp.ncbi.nlm.nih.gov/blast/executables/legacy.NOTSUPPORTED/2.2.26/blast-2.2.26-x64-linux.tar.gz -O blast-2.2.26.tar.gz
fi
tar xf blast-2.2.26.tar.gz

# download cs-blast
echo "downloading cs-blast . . ."
if [ "$SYS" = "Darwin" ]
then
    wget http://wwwuser.gwdg.de/~compbiol/data/csblast/releases/csblast-2.2.3_macosx.tar.gz -O csblast-2.2.3.tar.gz
else
    wget http://wwwuser.gwdg.de/~compbiol/data/csblast/releases/csblast-2.2.3_linux64.tar.gz -O csblast-2.2.3.tar.gz
fi
mkdir -p csblast-2.2.3
tar xf csblast-2.2.3.tar.gz -C csblast-2.2.3 --strip-components=1

# download and install gnu-parallel
echo "downloading gnu-parallel . . ."
wget https://ftpmirror.gnu.org/parallel/parallel-latest.tar.bz2
mkdir -p parallel
tar xf parallel-latest.tar.bz2 -C parallel --strip-components=1
cd parallel
./configure --prefix=`pwd` && make && make install
cd ..
