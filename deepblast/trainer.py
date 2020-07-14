import datetime
import argparse
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
import pytorch_lightning as pl
from deepblast.alignment import NeedlemanWunschAligner
from deepblast.dataset.alphabet import UniprotTokenizer
from deepblast.dataset import TMAlignDataset
from deepblast.dataset.dataset import decode, states2edges, collate_f
from deepblast.losses import (
    SoftAlignmentLoss, SoftPathLoss, MatrixCrossEntropy)
from deepblast.score import roc_edges, alignment_visualization, alignment_text
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence


class LightningAligner(pl.LightningModule):

    def __init__(self, args):
        super(LightningAligner, self).__init__()
        self.tokenizer = UniprotTokenizer()
        self.hparams = args
        self.initialize_aligner()
        if self.hparams.loss == 'sse':
            self.loss_func = SoftAlignmentLoss()
        if self.hparams.loss == 'cross_entropy':
            self.loss_func = MatrixCrossEntropy()
        elif self.hparams.loss == 'path':
            self.loss_func = SoftPathLoss()
        else:
            raise ValueError(f'{args.loss} is not implemented.')

    def initialize_aligner(self):
        n_alpha = len(self.tokenizer.alphabet)
        n_embed = self.hparams.embedding_dim
        n_input = self.hparams.rnn_input_dim
        n_units = self.hparams.rnn_dim
        n_layers = self.hparams.layers
        if self.hparams.aligner == 'nw':
            self.aligner = NeedlemanWunschAligner(
                n_alpha, n_input, n_units, n_embed, n_layers)
        else:
            raise NotImplementedError(
                f'Aligner {self.hparams.aligner_type} not implemented.')

    def forward(self, x, y):
        return self.aligner.forward(x, y)

    def initialize_logging(self, root_dir='./', logging_path=None):
        if logging_path is None:
            basename = "logdir"
            suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
            logging_path = "_".join([basename, suffix])
        full_path = root_dir + logging_path
        writer = SummaryWriter(full_path)
        return writer

    def train_dataloader(self):
        train_dataset = TMAlignDataset(
            self.hparams.train_pairs,
            construct_paths=isinstance(self.loss_func, SoftPathLoss))
        train_dataloader = DataLoader(
            train_dataset, self.hparams.batch_size, collate_fn=collate_f,
            shuffle=True, num_workers=self.hparams.num_workers,
            pin_memory=False)
        return train_dataloader

    def val_dataloader(self):
        valid_dataset = TMAlignDataset(
            self.hparams.valid_pairs,
            construct_paths=isinstance(self.loss_func, SoftPathLoss))
        valid_dataloader = DataLoader(
            valid_dataset, self.hparams.batch_size, collate_fn=collate_f,
            shuffle=False, num_workers=self.hparams.num_workers,
            pin_memory=False)
        return valid_dataloader

    def test_dataloader(self):
        test_dataset = TMAlignDataset(
            self.hparams.test_pairs,
            construct_paths=isinstance(self.loss_func, SoftPathLoss))
        test_dataloader = DataLoader(
            test_dataset, self.hparams.batch_size, shuffle=False,
            collate_fn=collate_f, num_workers=self.hparams.num_workers,
            pin_memory=False)
        return test_dataloader

    def compute_loss(self, x, y, predA, A, P, theta):

        if isinstance(self.loss_func, SoftAlignmentLoss):
            loss = self.loss_func(A, predA, x, y)
        elif isinstance(self.loss_func, MatrixCrossEntropy):
            loss = self.loss_func(A, predA, x, y)
        elif isinstance(self.loss_func, SoftPathLoss):
            loss = self.loss_func(P, predA, x, y)

        if self.hparams.multitask:
            current_lr = self.trainer.lr_schedulers[0].get_lr()[0]
            max_lr = self.hparams.learning_rate
            lam = current_lr / max_lr
            match_loss = self.loss_func(theta, predA, x, y)
            # when learning rate is large, weight match
            # otherwise, weight towards DP
            loss = lam * match_loss + (1 - lam) * loss
        return loss

    def training_step(self, batch, batch_idx):
        self.aligner.train()
        x, y, s, A, P = batch
        x = pack_sequence(x, enforce_sorted=False)
        y = pack_sequence(y, enforce_sorted=False)
        predA, theta = self.aligner(x, y)
        loss = self.compute_loss(x, y, predA, A, P, theta)
        assert torch.isnan(loss).item() is False
        current_lr = self.trainer.lr_schedulers[0].get_lr()[0]
        tensorboard_logs = {'train_loss': loss, 'lr': current_lr}
        # log the learning rate
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y, s, A, P = batch
        x = pack_sequence(x, enforce_sorted=False)
        y = pack_sequence(y, enforce_sorted=False)
        predA, theta = self.aligner(x, y)
        loss = self.compute_loss(x, y, predA, A, P, theta)
        # assert torch.isnan(loss).item() is False
        # Obtain alignment statistics + visualizations
        gen = self.aligner.traceback(x, y)
        statistics = []
        x, xlen = pad_packed_sequence(x, batch_first=True)
        y, ylen = pad_packed_sequence(y, batch_first=True)
        for b in range(len(s)):
            # TODO: Issue #47
            x_str = decode(
                list(x[b, :xlen[b]].squeeze().cpu().detach().numpy()),
                self.tokenizer.alphabet)
            y_str = decode(
                list(y[b, :ylen[b]].squeeze().cpu().detach().numpy()),
                self.tokenizer.alphabet)
            decoded, pred_A = next(gen)
            pred_x, pred_y, pred_states = list(zip(*decoded))
            pred_states = list(pred_states)
            truth_states = list(s[b].cpu().detach().numpy())
            pred_edges = list(zip(pred_y, pred_x))
            true_edges = states2edges(truth_states)
            stats = roc_edges(true_edges, pred_edges)
            if random.random() < self.hparams.visualization_fraction:
                try:
                    # TODO: Need to figure out wtf is happening here.
                    # See issue #40
                    text = alignment_text(
                        x_str, y_str, pred_states, truth_states)
                    self.logger.experiment.add_text(
                        f'alignment/{batch_idx}/{b}', text, self.global_step)
                except:
                    continue
                fig, _ = alignment_visualization(
                    A[b].cpu().detach().numpy().squeeze(),
                    predA[b].cpu().detach().numpy().squeeze(),
                    xlen[b], ylen[b])
                self.logger.experiment.add_figure(
                    f'alignment-matrix/{batch_idx}/{b}', fig, self.global_step)
            statistics.append(stats)
        statistics = pd.DataFrame(
            statistics, columns=[
                'val_tp', 'val_fp', 'val_fn', 'val_perc_id',
                'val_ppv', 'val_fnr', 'val_fdr'
            ]
        )
        statistics = statistics.mean(axis=0).to_dict()
        tensorboard_logs = {'valid_loss': loss}
        tensorboard_logs = {**tensorboard_logs, **statistics}
        return {'validation_loss': loss,
                'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        loss_f = lambda x: x['validation_loss']
        losses = list(map(loss_f, outputs))
        loss = sum(losses) / len(losses)
        self.logger.experiment.add_scalar('val_loss', loss, self.global_step)
        metrics = ['val_tp', 'val_fp', 'val_fn', 'val_perc_id',
                   'val_ppv', 'val_fnr', 'val_fdr']
        scores = []
        for i, m in enumerate(metrics):
            loss_f = lambda x: x['log'][m]
            losses = list(map(loss_f, outputs))
            scalar = sum(losses) / len(losses)
            scores.append(scalar)
            self.logger.experiment.add_scalar(m, scalar, self.global_step)
        tensorboard_logs = dict(
            [('val_loss', loss)] + list(zip(metrics, scores))
        )
        return {'val_loss': loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        pass

    def configure_optimizers(self):
        for p in self.aligner.lm.parameters():
            p.requires_grad = False
        grad_params = list(filter(
            lambda p: p.requires_grad, self.aligner.parameters()))
        optimizer = torch.optim.Adam(
            grad_params, lr=self.hparams.learning_rate)
        if self.hparams.scheduler == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=100, T_mult=2)
        elif self.hparams.scheduler == 'steplr':
            m = 1e-8  # minimum learning rate
            steps = int(np.log2(self.hparams.learning_rate / m))
            steps = self.hparams.epochs // steps
            scheduler = StepLR(optimizer, step_size=steps)
        else:
            s = self.hparams.scheduler
            raise ValueError(f'{s} is not implemented.')
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument(
            '--train-pairs', help='Training pairs file', required=True)
        parser.add_argument(
            '--test-pairs', help='Testing pairs file', required=True)
        parser.add_argument(
            '--valid-pairs', help='Validation pairs file', required=True)
        parser.add_argument(
            '-a', '--aligner',
            help='Aligner type. Choices include (nw, hmm).',
            required=False, type=str, default='nw')
        parser.add_argument(
            '--embedding-dim', help='Embedding dimension (default 512).',
            required=False, type=int, default=512)
        parser.add_argument(
            '--rnn-input-dim', help='RNN input dimension (default 512).',
            required=False, type=int, default=512)
        parser.add_argument(
            '--rnn-dim', help='Number of hidden RNN units (default 512).',
            required=False, type=int, default=512)
        parser.add_argument(
            '--layers', help='Number of RNN layers (default 2).',
            required=False, type=int, default=2)
        parser.add_argument(
            '--loss',
            help=('Loss function. Options include {sse, path, cross_entropy} '
                  '(default path)'),
            default='path', required=False, type=str)
        parser.add_argument(
            '--learning-rate', help='Learning rate',
            required=False, type=float, default=5e-5)
        parser.add_argument(
            '--batch-size', help='Training batch size',
            required=False, type=int, default=32)
        parser.add_argument(
            '--multitask', default=False, required=False, type=bool,
            help='Compute multitask loss between DP and matchings'
        )
        parser.add_argument(
            '--finetune', help='Perform finetuning',
            default=False, required=False, type=bool)
        parser.add_argument(
            '--scheduler',
            help=('Learning rate scheduler '
                  '(choices include `cosine` and `steplr`'),
            default='cosine', required=False, type=str)
        parser.add_argument(
            '--epochs', help='Training batch size',
            required=False, type=int, default=10)
        parser.add_argument(
            '--visualization-fraction',
            help='Fraction of alignments to be visualized per epoch',
            required=False, type=float, default=0.1)
        parser.add_argument(
            '-o', '--output-directory',
            help='Output directory of model results', required=True)
        return parser
