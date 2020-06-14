import argparse
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import glob
import os
import re
from Bio import SeqIO

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from deepblast.train import LightningAlignment


def main(args):
    model = LightningAligner(args)
    # profiler = AdvancedProfiler()

    trainer = Trainer(
        max_nb_epochs=args.epochs,
        gpus=args.gpus,
        nb_gpu_nodes=args.nodes,
        accumulate_grad_batches=args.grad_accum,
        distributed_backend=args.backend,
        precision=args.precision,
        check_val_every_n_epoch=0.1,
        # profiler=profiler,
        # fast_dev_run=True
        #auto_scale_batch_size='power'
    )

    ckpt_path = os.path.join(
        args.output_directory,
        trainer.logger.name,
        f"version_{trainer.logger.version}",
        "checkpoints",
    )
    print(f'{ckpt_path}:', ckpt_path)
    # initialize Model Checkpoint Saver
    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        period=1,
        monitor='val_loss',
        mode='min',
        verbose=True
    )
    trainer.checkpoint_callback = checkpoint_callback
    trainer.fit(model)

    # In case this doesn't checkpoint
    torch.save(model.state_dict(),
               args.output_directory + '/model_current.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=int, default=None)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument('--backend', type=str, default=None)
    # options include ddp_cpu, dp, dpp

    parser = LightningAligner.add_model_specific_args(parser)
    hparams = parser.parse_args()
    main(hparams)
