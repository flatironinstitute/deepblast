#!/usr/bin/env python3

import argparse
import os
import sys
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from deepblast.trainer import DeepBLAST

def main(args):
    print('args', args)
    # if args.load_from_checkpoint is not None:
    #     model = DeepBLAST.load_from_checkpoint(
    #         args.load_from_checkpoint)
    # else:
    model = DeepBLAST(**vars(args))
    # profiler = AdvancedProfiler()

    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_directory,
        monitor='validation_loss',
        filename="{epoch}-{step}-{validation_loss:0.4f}",
        verbose=True
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        num_nodes=args.nodes,
        #accumulate_grad_batches=args.grad_accum,
        #gradient_clip_val=args.grad_clip,
        # distributed_backend=args.backend,
        # precision=args.precision,
        val_check_interval=0.25,
        fast_dev_run=False,
        gradient_clip_algorithm="norm",
        strategy=DDPStrategy(find_unused_parameters=False),
        callbacks=[checkpoint_callback]
    )


    print('model', model)
    trainer.fit(model)
    trainer.test()

    # In case this doesn't checkpoint
    torch.save(model.state_dict(),
               os.path.join(args.output_directory, 'last_ckpt.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--accelerator', type=str, default=None)
    parser.add_argument('--devices', type=int, default=None)
    #parser.add_argument('--grad-accum', type=int, default=1)
    #parser.add_argument('--grad-clip', type=int, default=0)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=1)
    # parser.add_argument('--precision', type=int, default=32)
    # parser.add_argument('--load-from-checkpoint', type=str, default=None)

    parser = DeepBLAST.add_model_specific_args(parser)
    hparams = parser.parse_args()
    main(hparams)
