#!/bin/bash

# save current run info so daemons can keep track of process
echo "$HOSTNAME $$ $0" > $1.run_CASP14_server

############################################################
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/software/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/software/conda/etc/profile.d/conda.sh" ]; then
        . "/software/conda/etc/profile.d/conda.sh"
    else
        export PATH="/software/conda/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
############################################################


# Inputs:
IN="$1"    # input.fasta
WDIR="$2"  # working folder
TARGET="$3"

LEN=`tail -n1 $IN | wc -m`

PIPE_DIR="../casp14"
CPU="4"
MEM="30"

NODE="short"
NODEGPU="gpu" # "gpu"  # gpu-interactive" #robetta-gpu"
NODEGPUBIG="gpu"

mkdir -p $WDIR/log

############################################################
# 1. generate MSAs
############################################################
if [ ! -s $WDIR/t000_.msa0.a3m ]
then
    echo "Running HHblits"
    srun -p $NODE -c $CPU --mem=64g \
        -o "$WDIR/log/make_msa-%j.stdout" \
        -e "$WDIR/log/make_msa-%j.stderr" \
        $PIPE_DIR/scripts/make_msa.sh $IN $WDIR $CPU $MEM
fi

############################################################
# 2. predict secondary structure
############################################################

# start fresh
rm $WDIR/t000_.ss2

if [ ! -s $WDIR/t000_.ss2 ]
then
    echo "Running PSIPRED"
    $PIPE_DIR/scripts/make_ss.sh $WDIR/t000_.msa0.a3m $WDIR/t000_.ss2 > $WDIR/log/make_ss.stdout 2> $WDIR/log/make_ss.stderr
fi

############################################################
# 3. search for templates
############################################################
DB="/projects/ml/TrRosetta/pdb100_2020Mar11/pdb100_2020Mar11"
if [ ! -s $WDIR/t000_.hhr ]
then
    echo "Running hhsearch"
    export HHLIB=/home/robetta/rosetta_server_beta/src/hhsuite-3.0-beta.3-Linux
    export PATH=$HHLIB/bin:$PATH
    HH="hhsearch -b 50 -B 500 -z 50 -Z 500 -mact 0.05 -cpu $CPU -maxmem $MEM -aliw 100000 -e 100 -p 5.0 -d $DB"
    
    cat $WDIR/t000_.ss2 $WDIR/t000_.msa0.a3m > $WDIR/t000_.msa0.ss2.a3m
     srun -p $NODE -c $CPU --mem=${MEM}g \
        -o "$WDIR/log/hhsearch-%j.stdout" \
        -e "$WDIR/log/hhsearch-%j.stderr" \
        $HH -i $WDIR/t000_.msa0.ss2.a3m -o $WDIR/t000_.hhr -v 0
 fi

############################################################
# 4. generate TAPE features
############################################################
if [ ! -s $WDIR/t000_.tape.npy ]
then
    echo "Generating TAPE features"
    conda activate /home/robetta/.conda/envs/tape
    python $PIPE_DIR/trRosetta/tape/get_embeddings.py $IN $WDIR/t000_.tape.npy
    conda deactivate
fi

############################################################
# 5. predict distances and orientations
############################################################

conda activate /software/conda/envs/tensorflow

# use single representative MSA instead of multiple MSAs
# run msa-net
if [ ! -s $WDIR/t000_.msa.npz ]
then
    if [ $LEN -gt 700 ]
    then
	echo "Running predict_discont_masked.py for large target"
        srun -p $NODEGPU -c $CPU --mem=${MEM}g --gres=gpu:rtx2080:1 -o "$WDIR/log/msa-net-%j.stdout" -e "$WDIR/log/msa-net-%j.stderr" -J "msa-net"\
            python $PIPE_DIR/trRosetta/msa-net/predict_discont_masked.py \
                --windowed \
                --roll \
                -m $PIPE_DIR/trRosetta/msa-net/network01_discont \
                $WDIR/t000_.msa0.a3m $WDIR/t000_.tape.npy $WDIR/t000_.msa.npz &
    else
	echo "Running predict.py"
        srun -p $NODEGPU -c $CPU --mem=${MEM}g --gres=gpu:rtx2080:1 -o "$WDIR/log/msa-net-%j.stdout" -e "$WDIR/log/msa-net-%j.stderr" -J "msa-net"\
            python $PIPE_DIR/trRosetta/msa-net/predict.py \
                --windowed \
                --roll \
                -m $PIPE_DIR/trRosetta/msa-net/network02_1 \
                $WDIR/t000_.msa0.a3m $WDIR/t000_.tape.npy $WDIR/t000_.msa.npz &
    fi
fi
# tbm-net
if [ ! -s $WDIR/t000_.tbm.npz ]
then
    if [ $LEN -gt 400 ]
    then
	echo "Running predict_discont.py for large target for t000_.tbm.npz"
        srun -p $NODEGPU -c $CPU --mem=${MEM}g --gres=gpu:rtx2080:1 -o "$WDIR/log/tbm-net-%j.stdout" -e "$WDIR/log/tbm-net-%j.stderr" -J "tbm-net"\
            python $PIPE_DIR/trRosetta/tbm-net/predict_discont.py \
                --windowed \
                --roll \
                -m $PIPE_DIR/trRosetta/tbm-net/network00_discont \
                -t $DB \
                -n 25 \
                $WDIR/t000_.msa0.a3m $WDIR/t000_.tape.npy $WDIR/t000_.hhr $WDIR/t000_.tbm.npz &
    else
	echo "Running predict.py for t000_.tbm.npz"
        srun -p $NODEGPU -c $CPU --mem=${MEM}g --gres=gpu:rtx2080:1 -o "$WDIR/log/tbm-net-%j.stdout" -e "$WDIR/log/tbm-net-%j.stderr" -J "tbm-net"\
            python $PIPE_DIR/trRosetta/tbm-net/predict.py \
                --windowed \
                --roll \
                -m $PIPE_DIR/trRosetta/tbm-net/network01 \
                -t $DB \
                -n 25 \
                $WDIR/t000_.msa0.a3m $WDIR/t000_.tape.npy $WDIR/t000_.hhr $WDIR/t000_.tbm.npz &
    fi
fi

# wait until all submitted jobs finish
wait

# file latency
if [ ! -s $WDIR/t000_.tbm.npz ]
then
    sleep 30s
fi
if [ ! -s $WDIR/t000_.msa.npz ]
then
    sleep 30s
fi

############################################################
# 6. perform modeling
############################################################
conda deactivate
conda activate /software/conda/envs/pyrosetta

if [ ! -f $WDIR/DONE_iter0 ]
then
    
    mkdir -p $WDIR/pdb-msa
    mkdir -p $WDIR/pdb-tbm
    
    for m in 0 1 2
    do
        for p in 0.05 0.15 0.25 0.35 0.45
        do
            #for ((i=0;i<3;i++))   # -dk speed up for cameo testing
            for ((i=0;i<1;i++))
            do
                echo "python -u $PIPE_DIR/folding/RosettaTR.py -r 3 -pd $p -m $m -sg 7,3 $WDIR/t000_.msa.npz $IN $WDIR/pdb-msa/model${i}_${m}_${p}.pdb"
                echo "python -u $PIPE_DIR/folding/RosettaTR.py -r 3 -pd $p -m $m -sg 7,3 $WDIR/t000_.tbm.npz $IN $WDIR/pdb-tbm/model${i}_${m}_${p}.pdb"
            done
        done
    done > $WDIR/parallel.list
    
    N=`cat $WDIR/parallel.list | wc -l`
    echo "Running parallel RosettaTR.py"    
    sbatch -a 1-$N --wait -p $NODE --mem=6g -J $TARGET.fold\
        -o $WDIR/log/folding.stdout -e $WDIR/log/folding.stderr \
        --wrap="eval \`head -n \$SLURM_ARRAY_TASK_ID $WDIR/parallel.list | tail -1\`"
   
    touch $WDIR/DONE_iter0
    # file latency
    sleep 30s
fi

############################################################
# 7. Run trRefine
############################################################

conda deactivate
conda activate /software/conda/envs/tensorflow

# start fresh
rm $WDIR/t000_.trRefine.npz
rm -rf $WDIR/rep_s


if [ ! -s $WDIR/t000_.trRefine.npz ]
then
    if [ $LEN -gt 700 ]
    then
        echo "Running long sequence run_trRefine_DAN.py"
        srun -p $NODEGPUBIG -c $CPU --mem=${MEM}g --gres=gpu:titan:1 -o "$WDIR/log/trRefine-%j.stdout" -e "$WDIR/log/trRefine-%j.stderr" -J "trRefine" \
            python $PIPE_DIR/trRefine/run_trRefine_DAN.py -msa_npz $WDIR/t000_.msa.npz \
            -tbm_npz $WDIR/t000_.tbm.npz -pdb_dir_s $WDIR/pdb-msa $WDIR/pdb-tbm \
            -a3m_fn $WDIR/t000_.msa0.a3m -hhr_fn $WDIR/t000_.hhr -n_core $CPU
    else
        echo "Running run_trRefine_DAN.py"
        srun -p $NODEGPU -c $CPU --mem=${MEM}g --gres=gpu:rtx2080:1 -o "$WDIR/log/trRefine-%j.stdout" -e "$WDIR/log/trRefine-%j.stderr" -J "trRefine" \
            python $PIPE_DIR/trRefine/run_trRefine_DAN.py -msa_npz $WDIR/t000_.msa.npz \
            -tbm_npz $WDIR/t000_.tbm.npz -pdb_dir_s $WDIR/pdb-msa $WDIR/pdb-tbm \
            -a3m_fn $WDIR/t000_.msa0.a3m -hhr_fn $WDIR/t000_.hhr -n_core $CPU
    fi
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
            #for ((i=0;i<3;i++))   # -dk speed up for cameo testing
            for ((i=0;i<1;i++))
            do
                echo "python -u $PIPE_DIR/folding/RosettaTR.py -r 3 -pd $p -m $m -sg 7,3 -bb $WDIR/rep_s/BBtor.npz $WDIR/t000_.trRefine.npz $IN $WDIR/pdb-trRefine/model${i}_${m}_${p}.pdb"
            done
        done
    done > $WDIR/trRefine_fold.list
    
    N=`cat $WDIR/trRefine_fold.list | wc -l`
    echo "Running parallel RosettaTR.py refinement" 
    sbatch -a 1-$N --wait -p $NODE --mem=6g -J $TARGET.fold \
        -o $WDIR/log/trRef_fold.stdout -e $WDIR/log/trRef_fold.stderr \
        --wrap="eval \`head -n \$SLURM_ARRAY_TASK_ID $WDIR/trRefine_fold.list | tail -1\`"
   
    touch $WDIR/DONE_iter1
    # file latency
    sleep 30s
    ls $WDIR/pdb-trRefine/model*.pdb > $WDIR/pdb-trRefine/pdb.list
fi

############################################################
# 9. Pick final models
############################################################

if [ ! -f $WDIR/pdb-trRefine/DONE_DAN ]
then
    # run DeepAccNet-msa
    echo "Running DeepAccNet-msa ErrorPredictorMSA.py"
    srun -p $NODEGPU -c $CPU --mem=${MEM}g --gres=gpu:rtx2080:1 -o "$WDIR/log/DAN_fin-%j.stdout" -e "$WDIR/log/DAN_fin-%j.stderr" -J "DAN-msa"\
        python $PIPE_DIR/trRefine/DAN-msa/ErrorPredictorMSA.py \
            -p 5 \
            $WDIR/t000_.trRefine.npz $WDIR/pdb-trRefine/pdb.list $WDIR/pdb-trRefine
    touch $WDIR/pdb-trRefine/DONE_DAN
    # file latency
    sleep 30s
fi

if [ ! -s $WDIR/model/model_5.crderr.pdb ]
then
    echo "Running pick_final_models.div.py"
    srun -p $NODE -c 5 --mem=${MEM}g -o "$WDIR/log/final-%j.stdout" -e "$WDIR/log/final-%j.stderr" -J $TARGET.final\
        python -u -W ignore $PIPE_DIR/trRefine/pick_final_models.div.py \
            $WDIR/pdb-trRefine $WDIR/rep_s $WDIR/model 5
    # file latency
    sleep 30s
fi
echo "Done"

