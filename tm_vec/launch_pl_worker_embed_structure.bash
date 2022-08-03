#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH -N2
#SBATCH --mem 700gb
#SBATCH --tasks-per-node=4
#SBATCH --gres=gpu:v100-32gb:04
#SBATCH --job-name=train
#SBATCH -o slurm-%x.%j.out


module load slurm
source /mnt/home/thamamsy/software/conda/bin/activate
conda activate deep_blast_env

export METAG_NNODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
echo "Number of nodes: $METAG_NNODES"

DATA_DIR=/mnt/home/thamamsy/ceph/deepblast/data
#DATA_DIR=/mnt/home/thamamsy/ceph/cath/cath/non_redundant
export METAG_DATA=${DATA_DIR}/tm_protrans_for_model_medium.pickle
#export METAG_DATA=${DATA_DIR}/pairs_cath_250_tm_protrans_embeddings.gz
export METAG_RANDOM_SEED=$RANDOM int = 10
export METAG_DMODEL=1024
export METAG_NLAYER=2
export METAG_NHEADS=4
export METAG_IN_DIM=2048
export METAG_WARMUP_STEPS=300
export METAG_TRAIN_PROP=0.90 
export METAG_VAL_PROP=0.05
export METAG_TEST_PROP=0.05

set -ex
export METAG_LR=0.0001
export METAG_BSIZE=16
export METAG_SESSION=/mnt/home/thamamsy/ceph/deepblast/models/transformer_lr${METAG_LR}_dmodel${METAG_DMODEL}_nlayer${METAG_NLAYER}_cosine_sigmoid_big_data
export METAG_RANDOM_SEED=$RANDOM
export EPOCHS=20


set +x

srun pl_worker_embed_structure.bash
