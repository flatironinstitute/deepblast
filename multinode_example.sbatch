#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH -N2
#SBATCH --cpus-per-task 10
#SBATCH --tasks-per-node=4
#SBATCH --gres=gpu:v100-32gb:4
#SBATCH --job-name=train
#SBATCH -o slurm-%x.%j.out

module -q purge
module -q load modules/2.1-testing cuda cudnn python3

# You should change this...
source ~rblackwell/envs/deepblast/bin/activate
DBROOT=/mnt/home/rblackwell/projects/codes/deepblast


export METAG_NNODES=$SLURM_JOB_NUM_NODES
echo "Number of nodes: $METAG_NNODES"

set -ex
METAG_LR=0.0001
METAG_BATCH_SIZE=16
METAG_EPOCHS=20

# You should put your data on ceph, not $HOME
METAG_output_directory=$DBROOT/test/${METAG_LR}_output
METAG_TRAIN_DATA=$DBROOT/data/tm_align_output_10k.tab
METAG_VAL_DATA=$DBROOT/data/tm_align_output_10k.tab
METAG_TEST_DATA=$DBROOT/data/tm_align_output_10k.tab

# Some bug right now with multiple workers in this configuration, so we actually only want 1 worker for now
# NWORKERS=$((SLURM_CPUS_PER_TASK))
NWORKERS=1

# nvidia-smi doesn't work if using salloc, since the script runs on
# the submitting machine, not the first allocated machine
NGPUS=${SLURM_NTASKS_PER_NODE}

# Occupancy isn't the main bottleneck. Numba wants larger grid sizes,
# but that ends up being slower. Tell it to shut up about it.
export NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0

# If submitting from a workstation, it will try to download to a
# non-existent directory, so don't do that.
export XDG_CACHE_HOME=$HOME/.cache

set +x

srun python3 $DBROOT/deepblast/deepblast_train.py \
     --nodes ${METAG_NNODES} \
     --accelerator 'gpu' \
     --devices ${SLURM_NTASKS_PER_NODE} \
     --num-workers ${NWORKERS} \
     --train-pairs ${METAG_TRAIN_DATA} \
     --valid-pairs ${METAG_VAL_DATA} \
     --test-pairs ${METAG_TEST_DATA} \
     --learning-rate ${METAG_LR} \
     --batch-size ${METAG_BATCH_SIZE} \
     --epochs ${METAG_EPOCHS} \
     --output-directory ${METAG_output_directory}
