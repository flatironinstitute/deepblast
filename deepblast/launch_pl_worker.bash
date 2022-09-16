#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH -N1
#SBATCH --mem 700gb
#SBATCH --tasks-per-node=4
#SBATCH --gres=gpu:v100-32gb:04
#SBATCH --job-name=train
#SBATCH -o slurm-%x.%j.out


module load slurm
module load cuda/11.4.4 cudnn/8.2.4.15-11.4 gcc/11.2.0 

source /mnt/home/thamamsy/software/conda/bin/activate
conda activate huggingface_meta #deep_blast_env

export METAG_NNODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
echo "Number of nodes: $METAG_NNODES"


set -ex
export METAG_LR=0.0001
export METAG_BATCH_SIZE=16
export METAG_EPOCHS=20
export METAG_output_directory=/mnt/home/thamamsy/ceph/deepblast/models/deepblast_esm_${METAG_LR}_test
export METAG_TRAIN_DATA=/mnt/home/thamamsy/projects/deep_blast_pull/deepblast/data/tm_align_output_10k.tab
export METAG_VAL_DATA=/mnt/home/thamamsy/projects/deep_blast_pull/deepblast/data/tm_align_output_10k.tab
export METAG_TEST_DATA=/mnt/home/thamamsy/projects/deep_blast_pull/deepblast/data/tm_align_output_10k.tab


set +x

srun pl_worker.bash
