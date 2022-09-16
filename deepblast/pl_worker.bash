#!/usr/bin/env sh

##################################################################

NGPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
module load gcc/7.5.0 cuda/11.4.2 cudnn/8.2.4.15-11.4

#export NCCL_SOCKET_IFNAME=$(./scripts/get_ifname.py)
export NODE_RANK=${SLURM_NODEID}

python deepblast_train2.py \
    --nodes ${METAG_NNODES} \
    --gpus ${NGPUS} \
    --train-pairs ${METAG_TRAIN_DATA} \
    --valid-pairs ${METAG_VAL_DATA} \
    --test-pairs ${METAG_TEST_DATA} \
    --learning-rate ${METAG_LR} \
    --batch-size ${METAG_BATCH_SIZE} \
    --epochs ${METAG_EPOCHS} \
    --output-directory ${METAG_output_directory} 

