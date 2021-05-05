#!/bin/bash

# resources
CPU="$3"
MEM="$4"
echo $CPU $MEM

# setup hhblits
export HHLIB=/home/robetta/rosetta_server_beta/src/hhsuite-3.0-beta.3-Linux
export PATH=$HHLIB/bin:$PATH
DB="/local/robetta/local_db_nrb/uniclust/uniref30_2020_01/UniRef30_2020_01"
if [ ! -s /local/robetta/local_db_nrb/uniclust/uniref30_2020_01/UniRef30_2020_01_a3m.ffdata ]
then
    DB="/databases/uniclust/30/2020_01/UniRef30_2020_01"
fi
HHBLITS="hhblits -o /dev/null -mact 0.35 -maxfilt 20000 -neffmax 20 -cpu $CPU -nodiff -realign_max 20000 -maxmem $MEM -n 4 -d $DB"

# inputs
in_fasta="$1"
out_dir="$2"

mkdir -p $out_dir/hhblits
tmp_dir="$out_dir/hhblits"
out_prefix="$out_dir/t000_"

# perform iterative searches
prev_a3m="$in_fasta"
#for e in 1e-80 1e-70 1e-60 1e-50 1e-40 1e-30 1e-20 1e-10 1e-8 1e-6 1e-4 1e-3 1e-1
#for e in 1e-30 1e-10 1e-6 1e-3 1e-1   # -dk speed up for cameo testing
for e in 1e-30 1e-10 1e-6 1e-3   # -minkyung speed up for cameo testing. it'll use representative MSA (msa0) only (Nov 12, 2020)
do
    echo $e
    $HHBLITS -i $prev_a3m -oa3m $tmp_dir/t000_.$e.a3m -e $e -v 0
    hhfilter -id 90 -cov 75 -maxseq 10000 -i $tmp_dir/t000_.$e.a3m -o $tmp_dir/t000_.$e.id90cov75.a3m
    hhfilter -id 90 -cov 50 -maxseq 10000 -i $tmp_dir/t000_.$e.a3m -o $tmp_dir/t000_.$e.id90cov50.a3m
    prev_a3m="$tmp_dir/t000_.$e.id90cov50.a3m"
    n75=`grep -c "^>" $tmp_dir/t000_.$e.id90cov75.a3m`
    n50=`grep -c "^>" $tmp_dir/t000_.$e.id90cov50.a3m`

    if ((n75>2000))
    then
        if [ ! -s ${out_prefix}.msa0.a3m ]
        then
            cp $tmp_dir/t000_.$e.id90cov75.a3m ${out_prefix}.msa0.a3m
	    break # -minkyung add this (Nov 12, 2020)
        fi
    elif ((n50>5000))
    then
        if [ ! -s ${out_prefix}.msa0.a3m ]
        then
            cp $tmp_dir/t000_.$e.id90cov50.a3m ${out_prefix}.msa0.a3m
	    break # -minkyung add this (Nov 12, 2020)
        fi
    else
        continue
    fi

done

if [ ! -s ${out_prefix}.msa0.a3m ]
then
    cp $tmp_dir/t000_.1e-3.id90cov50.a3m ${out_prefix}.msa0.a3m
fi

# -minkyung commented the following scripts because only representative MSA (msa0) will be used for the rest of the part (Nov 12, 2020)
#i=1
##for e in 1e-80 1e-40 1e-10 1e-3 1e-1  # -dk speed up for cameo testing
#for e in 1e-3 1e-1
#do
#    hhfilter -id 95 -i $tmp_dir/t000_.$e.a3m -o ${out_prefix}.msa$i.a3m
#    ((i=i+1))
#done

#rm -r $tmp_dir
