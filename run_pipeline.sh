#!/bin/bash

SCRIPT=`realpath -s $0`
export PIPEDIR=`dirname $SCRIPT`

CPU="4"  # number of CPUs to use
MEM="32" # max memory (in GB)

# Inputs:
IN="$1"                # input.fasta
WDIR=`realpath -s $2`  # working folder


LEN=`tail -n1 $IN | wc -m`

mkdir -p $WDIR/log

############################################################
# 1. generate MSAs
############################################################
if [ ! -s $WDIR/t000_.msa0.a3m ]
then
    echo "Running HHblits"
    $PIPEDIR/scripts/make_msa.sh $IN $WDIR $CPU $MEM > $WDIR/log/make_msa.stdout 2> $WDIR/log/make_msa.stderr
fi


############################################################
# 2. predict secondary structure
############################################################
if [ ! -s $WDIR/t000_.ss2 ]
then
    echo "Running PSIPRED"
    $PIPEDIR/scripts/make_ss.sh $WDIR/t000_.msa0.a3m $WDIR/t000_.ss2 > $WDIR/log/make_ss.stdout 2> $WDIR/log/make_ss.stderr
fi


############################################################
# 3. search for templates
############################################################
DB="/projects/ml/TrRosetta/pdb100_2020Mar11/pdb100_2020Mar11"
if [ ! -s $WDIR/t000_.hhr ]
then
    echo "Running hhsearch"
    DB="$PIPEDIR/pdb100_2020Mar11/pdb100_2020Mar11"
    HH="hhsearch -b 50 -B 500 -z 50 -Z 500 -mact 0.05 -cpu $CPU -maxmem $MEM -aliw 100000 -e 100 -p 5.0 -d $DB"
    cat $WDIR/t000_.ss2 $WDIR/t000_.msa0.a3m > $WDIR/t000_.msa0.ss2.a3m
    $HH -i $WDIR/t000_.msa0.ss2.a3m -o $WDIR/t000_.hhr -v 0 > $WDIR/log/hhsearch.stdout 2> $WDIR/log/hhsearch.stderr
fi


############################################################
# 4. generate TAPE features
############################################################
if [ ! -s $WDIR/t000_.tape.npy ]
then
    echo "Generating TAPE features"
    python $PIPEDIR/tape/get_embeddings.py $IN $WDIR/t000_.tape.npy
fi


############################################################
# 5. predict distances and orientations
############################################################

if [ $LEN -gt 700 ]
then
    crop="discont"
else
    crop="cont"
fi

# run msa-net
if [ ! -s $WDIR/t000_.msa.npz ]
then
    echo "Running sequence-based trRosetta"
    python $PIPEDIR/trRosetta/predict.py \
        -m $PIPEDIR/weights \
        -i $WDIR/t000_.msa0.a3m \
        -o $WDIR/t000_.msa.npz \
        --tape $WDIR/t000_.tape.npy \
        --crop $crop > $WDIR/log/msa-net.stdout 2> $WDIR/log/msa-net.stderr
fi

# tbm-net
if [ ! -s $WDIR/t000_.tbm.npz ]
then
    python $PIPEDIR/trRosetta/predict.py \
        -m $PIPEDIR/weights \
        -i $WDIR/t000_.msa0.a3m \
        -o $WDIR/t000_.tbm.npz \
        --tape $WDIR/t000_.tape.npy \
        --hhr $WDIR/t000_.hhr \
        --crop $crop > $WDIR/log/tbm-net.stdout 2> $WDIR/log/tbm-net.stderr
fi


############################################################
# 6. perform modeling
############################################################
if [ ! -f $WDIR/DONE_iter0 ]
then
    
    mkdir -p $WDIR/pdb-msa
    mkdir -p $WDIR/pdb-tbm
    
    for m in 0 1 2
    do
        for p in 0.05 0.15 0.25 0.35 0.45
        do
            for ((i=0;i<1;i++))
            do
                echo "python -u $PIPEDIR/folding/RosettaTR.py -r 3 -pd $p -m $m -sg 7,3 $WDIR/t000_.msa.npz $IN $WDIR/pdb-msa/model${i}_${m}_${p}.pdb"
                echo "python -u $PIPEDIR/folding/RosettaTR.py -r 3 -pd $p -m $m -sg 7,3 $WDIR/t000_.tbm.npz $IN $WDIR/pdb-tbm/model${i}_${m}_${p}.pdb"
            done
        done
    done > $WDIR/parallel.list
    echo "Folding trRosetta models"
    parallel -j $CPU < $WDIR/parallel.list > $WDIR/log/folding.stdout 2> $WDIR/log/folding.stderr
    touch $WDIR/DONE_iter0
fi


############################################################
# 7. Run trRefine
############################################################
if [ ! -s $WDIR/t000_.trRefine.npz ]
then
    echo "Running trRefine"
    cd $WDIR
    python $PIPEDIR/trRefine/run_trRefine_DAN.py -msa_npz $WDIR/t000_.msa.npz \
        -tbm_npz $WDIR/t000_.tbm.npz -pdb_dir_s $WDIR/pdb-msa $WDIR/pdb-tbm \
        -a3m_fn $WDIR/t000_.msa0.a3m -hhr_fn $WDIR/t000_.hhr \
        -n_core $CPU > $WDIR/log/trRefine.stdout 2> $WDIR/log/trRefine.stderr
    cd -
fi


############################################################
# 8. Run modeling w/ trRefine output
############################################################

if [ ! -f $WDIR/DONE_iter1 ]
then
    mkdir -p $WDIR/pdb-trRefine
    
    for m in 0 1 2
    do
        for p in 0.05 0.15 0.25 0.35 0.45
        do
            #for ((i=0;i<3;i++))
            for ((i=0;i<1;i++))
            do
                echo "python -u $PIPEDIR/folding/RosettaTR.py -r 3 -pd $p -m $m -sg 7,3 -bb $WDIR/rep_s/BBtor.npz $WDIR/t000_.trRefine.npz $IN $WDIR/pdb-trRefine/model${i}_${m}_${p}.pdb"
            done
        done
    done > $WDIR/trRefine_fold.list
    
    echo "Folding trRefine models"
    parallel -j $CPU < $WDIR/trRefine_fold.list > $WDIR/log/trRefine_fold.stdout 2> $WDIR/log/trRefine_fold.stderr
    touch $WDIR/DONE_iter1
    ls $WDIR/pdb-trRefine/model*.pdb > $WDIR/pdb-trRefine/pdb.list
fi


############################################################
# 9. Pick final models
############################################################

if [ ! -f $WDIR/pdb-trRefine/DONE_DAN ]
then

    # run DeepAccNet-msa
    echo "Running DeepAccNet-msa on trRefine models"
    python $PIPEDIR/trRefine/DAN-msa/ErrorPredictorMSA.py \
        -p $CPU \
        $WDIR/t000_.trRefine.npz $WDIR/pdb-trRefine/pdb.list $WDIR/pdb-trRefine > $WDIR/log/dan-msa.stdout 2> $WDIR/log/dan-msa.stderr
    touch $WDIR/pdb-trRefine/DONE_DAN
fi

if [ ! -s $WDIR/model/model_5.crderr.pdb ]
then
    echo "Picking final models"
    python -u -W ignore $PIPEDIR/trRefine/pick_final_models.div.py \
        $WDIR/pdb-trRefine $WDIR/rep_s $WDIR/model $CPU > $WDIR/log/pick.stdout 2> $WDIR/log/pick.stderr
    echo "Final models saved in: $2/model"
fi
echo "Done"
