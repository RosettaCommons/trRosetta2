#!/bin/bash

# inputs
in_fasta="$1"
out_dir="$2"

CPU="$3"
MEM="$4"

#DB="$PIPEDIR/UniRef30_2020_06_hhsuite/UniRef30_2020_06"
DB="/databases/uniclust/30/latest/UniRef30_2020_06"

# search command
HHBLITS="hhblits -o /dev/null -mact 0.35 -maxfilt 20000 -neffmax 20 -cpu $CPU -nodiff -realign_max 20000 -maxmem $MEM -n 4 -d $DB"

mkdir -p $out_dir/hhblits
tmp_dir="$out_dir/hhblits"
out_prefix="$out_dir/t000_"

# perform iterative searches
prev_a3m="$in_fasta"
#for e in 1e-80 1e-70 1e-60 1e-50 1e-40 1e-30 1e-20 1e-10 1e-8 1e-6 1e-4 1e-3
for e in 1e-30 1e-10 1e-6 1e-3
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
	    break
        fi
    elif ((n50>5000))
    then
        if [ ! -s ${out_prefix}.msa0.a3m ]
        then
            cp $tmp_dir/t000_.$e.id90cov50.a3m ${out_prefix}.msa0.a3m
	    break
        fi
    else
        continue
    fi

done

if [ ! -s ${out_prefix}.msa0.a3m ]
then
    cp $tmp_dir/t000_.1e-3.id90cov50.a3m ${out_prefix}.msa0.a3m
fi

