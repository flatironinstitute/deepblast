#!/usr/bin/env sh

##################################################################

NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
#module load gcc/7.4.0 cuda/11.1.0_455.23.05 cudnn/v8.0.4-cuda-11.1 
module load gcc/7.5.0 cuda/11.4.2 cudnn/8.2.4.15-11.4

#export NCCL_SOCKET_IFNAME=$(./scripts/get_ifname.py)
export NODE_RANK=${SLURM_NODEID}

python train_embed_structure_model.py \
    --nodes ${METAG_NNODES} \
    --gpus ${NGPUS} \
    --session ${METAG_SESSION} \
    --data ${METAG_DATA} \
    --nodes ${METAG_NNODES} \
    --gpus ${NGPUS} \
    --lr0 ${METAG_LR} \
    --max-epochs ${EPOCHS} \
    --batch-size ${METAG_BSIZE} \
    --d_model ${METAG_DMODEL} \
    --num_layers ${METAG_NLAYER} \
    --dim_feedforward ${METAG_IN_DIM} \
    --nhead ${METAG_NHEADS} \
    --warmup_steps ${METAG_WARMUP_STEPS} \
    --train-prop ${METAG_TRAIN_PROP} \
    --val-prop ${METAG_VAL_PROP} \
    --test-prop ${METAG_TEST_PROP}