#!/bin/bash

ncbidir="/home/robetta/rosetta_server_beta/src/blast/bin"
sourcedir="/home/robetta/rosetta_server_beta/src/casp14"
execdir="$sourcedir/psipred4/bin"
datadir="$sourcedir/psipred4/data"

i_a3m="$1"
o_ss="$2"

ID=$(basename $i_a3m .a3m).tmp

/home/robetta/rosetta_server_beta/src/csblast-2.2.3/bin/csbuild -i $i_a3m -I a3m -D  /home/robetta/rosetta_server_beta/src/csblast-2.2.3/data/K4000.crf -o $ID.chk -O chk

head -n 2 $i_a3m > $ID.fasta
echo $ID.chk > $ID.pn
echo $ID.fasta > $ID.sn

$ncbidir/makemat -P $ID
$execdir/psipred $ID.mtx $datadir/weights.dat $datadir/weights.dat2 $datadir/weights.dat3 > $ID.ss
$execdir/psipass2 $datadir/weights_p2.dat 1 1.0 1.0 $i_a3m.csb.hhblits.ss2 $ID.ss > $ID.horiz

(
echo ">ss_pred"
grep "^Pred" $ID.horiz | awk '{print $2}'
echo ">ss_conf"
grep "^Conf" $ID.horiz | awk '{print $2}'
) | awk '{if(substr($1,1,1)==">") {print "\n"$1} else {printf "%s", $1}} END {print ""}' | sed "1d" > $o_ss

rm ${i_a3m}.csb.hhblits.ss2
rm $ID.*
