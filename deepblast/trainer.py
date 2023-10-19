import datetime
import argparse
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, CyclicLR)
import pytorch_lightning as pl
from deepblast.alignment import NeuralAligner
from deepblast.dataset import TMAlignDataset
from deepblast.dataset.utils import (
    decode, states2edges, collate_f, test_collate_f,
    unpack_sequences, pack_sequences, revstate_f,
    get_sequence
)
from deepblast.losses import (
    SoftAlignmentLoss, SoftPathLoss, MatrixCrossEntropy)
from deepblast.score import (roc_edges, alignment_visualization,
                             alignment_text, filter_gaps)


class DeepBLAST(pl.LightningModule):

    def __init__(self, batch_size=20,
                 hidden_dim=1024,
                 embedding_dim=1024,
                 epochs=32,
                 finetune=False,
                 layers=1,
                 dropout=0,
                 lm=None,
                 tokenizer=None,
                 learning_rate=0.0001,
                 loss='cross_entropy',
                 mask_gaps=False,
                 multitask=False,
                 num_workers=1,
                 output_directory=None,
                 scheduler='cosine',
                 test_pairs=None,
                 train_pairs=None,
                 valid_pairs=None,
                 visualization_fraction=1.0,
                 shuffle_validation=False,
                 device='gpu',
                 alignment_mode='needleman-wunsch'
    ):

        super(DeepBLAST, self).__init__()
        self.save_hyperparameters(ignore=['lm', 'tokenizer'])


        if self.hparams.loss == 'sse':
            self.loss_func = SoftAlignmentLoss()
        elif self.hparams.loss == 'cross_entropy':
            self.loss_func = MatrixCrossEntropy()
        elif self.hparams.loss == 'path':
            self.loss_func = SoftPathLoss()
        else:
            raise ValueError(f'`{self.hparams.loss}` is not implemented.')

        # initialize_aligner
        n_embed = self.hparams.embedding_dim
        n_input = self.hparams.hidden_dim
        n_units = self.hparams.hidden_dim
        n_layers = self.hparams.layers
        dropout = self.hparams.dropout
        self.aligner = NeuralAligner(
            n_input, n_units, n_embed, n_layers, dropout=dropout, lm=lm,
            alignment_mode=alignment_mode,
            device=device)

    def align(self, x, y):
        x_code = get_sequence(x, self.tokenizer)[0].to(self.device)
        y_code = get_sequence(y, self.tokenizer)[0].to(self.device)
        seq, order = pack_sequences([x_code], [y_code])
        gen = self.aligner.traceback(seq, order)
        decoded, _ = next(gen)
        pred_x, pred_y, pred_states = zip(*decoded)
        s = ''.join(list(map(revstate_f, pred_states)))
        return s

    def forward(self, x, order):
        """
        Parameters
        ----------
        x : PackedSequence
            Packed sequence object of proteins to align.
        order : np.array
            The origin order of the sequences

        Returns
        -------
        aln : torch.Tensor
            Alignment Matrix (dim B x N x M)
        mu : torch.Tensor
            Match scoring matrix
        g : torch.Tensor
            Gap scoring matrix
        """
        aln, mu, g = self.aligner.forward(x, order)
        return aln, mu, g

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
            tokenizer=self.tokenizer,
            construct_paths=isinstance(self.loss_func, SoftPathLoss))
        train_dataloader = DataLoader(
            train_dataset, self.hparams.batch_size, collate_fn=collate_f,
            shuffle=True, num_workers=self.hparams.num_workers,
            pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        valid_dataset = TMAlignDataset(
            self.hparams.valid_pairs,
            tokenizer=self.tokenizer,
            construct_paths=isinstance(self.loss_func, SoftPathLoss))
        valid_dataloader = DataLoader(
            valid_dataset, self.hparams.batch_size, collate_fn=collate_f,
            shuffle=self.hparams.shuffle_validation,
            num_workers=self.hparams.num_workers,
            pin_memory=True)
        return valid_dataloader

    def test_dataloader(self):
        test_dataset = TMAlignDataset(
            self.hparams.test_pairs, return_names=True,
            tokenizer=self.tokenizer,
            construct_paths=isinstance(self.loss_func, SoftPathLoss))
        test_dataloader = DataLoader(
            test_dataset, self.hparams.batch_size, shuffle=False,
            collate_fn=test_collate_f, num_workers=self.hparams.num_workers,
            pin_memory=True)
        return test_dataloader

    def compute_loss(self, x, y, predA, A, P, G, theta):

        if isinstance(self.loss_func, SoftAlignmentLoss):
            loss = self.loss_func(A, predA, x, y, G)
        elif isinstance(self.loss_func, MatrixCrossEntropy):
            loss = self.loss_func(A, predA, x, y, G)
        elif isinstance(self.loss_func, SoftPathLoss):
            loss = self.loss_func(P, predA, x, y, G)
        if self.hparams.multitask:
            current_lr = self.trainer.lr_schedulers[0]['scheduler']
            current_lr = current_lr.get_last_lr()[0]
            max_lr = self.hparams.learning_rate
            lam = current_lr / max_lr
            match_loss = self.loss_func(torch.sigmoid(theta), predA, x, y)
            # when learning rate is large, weight match loss
            # otherwise, weight towards DP
            loss = lam * match_loss + (1 - lam) * loss
        return loss

    def training_step(self, batch, batch_idx):
        self.aligner.train()
        genes, others, s, A, P, G, _, _ = batch
        seq, order = pack_sequences(genes, others)
        predA, theta, gap = self.aligner(seq, order)
        _, xlen, _, ylen = unpack_sequences(seq, order)
        loss = self.compute_loss(xlen, ylen, predA, A, P, G, theta)
        assert torch.isnan(loss).item() is False
        # if len(self.trainer.lr_schedulers) >= 1:
        #     current_lr = self.trainer.lr_schedulers[0]['scheduler']
        #     current_lr = current_lr.get_last_lr()[0]
        # else:
        #     current_lr = self.hparams.learning_rate
        tensorboard_logs = {'train_loss': loss}
        # log the learning rate
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_stats(self, x, y, xlen, ylen, gen,
                         states, A, predA, theta, gap, batch_idx):
        statistics = []
        for b in range(len(xlen)):
            # TODO: Issue #47
            x_str = decode(
                list(x[b, :xlen[b]].squeeze().cpu().detach().numpy()),
                self.tokenizer.get_vocab())
            y_str = decode(
                list(y[b, :ylen[b]].squeeze().cpu().detach().numpy()),
                self.tokenizer.get_vocab())
            decoded, _ = next(gen)
            pred_x, pred_y, pred_states = list(zip(*decoded))
            pred_states = np.array(list(pred_states))
            true_states = states[b].cpu().detach().numpy()
            pred_edges = states2edges(pred_states)
            true_edges = states2edges(true_states)
            pred_edges = filter_gaps(pred_states, pred_edges)
            true_edges = filter_gaps(true_states, true_edges)
            stats = roc_edges(true_edges, pred_edges)
            if random.random() < self.hparams.visualization_fraction:
                Av = A[b].cpu().detach().numpy().squeeze()
                pv = predA[b].cpu().detach().numpy().squeeze()
                tv = theta[b].cpu().detach().numpy().squeeze()
                gv = gap[b].cpu().detach().numpy().squeeze()
                fig, _ = alignment_visualization(
                    Av, pv, tv, gv, xlen[b], ylen[b])
                self.logger.experiment.add_figure(
                    f'alignment-matrix/{batch_idx}/{b}', fig,
                    self.global_step, close=True)
                try:
                    text = alignment_text(
                        x_str, y_str, pred_states, true_states, stats)
                    self.logger.experiment.add_text(
                        f'alignment/{batch_idx}/{b}', text, self.global_step)
                except Exception as e:
                    print('predA', predA[b].shape)
                    print('A', A[b].shape)
                    print('theta', theta[b].shape)
                    print('xlen', xlen[b], 'ylen', ylen[b])
                    print('pred_states', pred_states, len(pred_states))
                    print('true_states', true_states, len(true_states))
            statistics.append(stats)
        return statistics

    def validation_step(self, batch, batch_idx):
        genes, others, s, A, P, G, _, _ = batch
        seq, order = pack_sequences(genes, others)
        predA, theta, gap = self.aligner(seq, order)
        x, xlen, y, ylen = unpack_sequences(seq, order)
        loss = self.compute_loss(xlen, ylen, predA, A, P, G, theta)

        assert torch.isnan(loss).item() is False

        self.log('validation_loss', loss,
                 batch_size=self.hparams.batch_size, sync_dist=True)
        tensorboard_logs = {'valid_loss': loss}

        # Obtain alignment statistics + visualizations
        if self.hparams.visualization_fraction > 0:
            gen = self.aligner.traceback(seq, order)
            # TODO; compare the traceback and the forward
            statistics = self.validation_stats(
                x, y, xlen, ylen, gen, s, A, predA, theta, gap, batch_idx)
            statistics = pd.DataFrame(
                statistics, columns=[
                    'val_tp', 'val_fp', 'val_fn', 'val_perc_id',
                    'val_ppv', 'val_fnr', 'val_fdr'
                ]
            )
            statistics = statistics.mean(axis=0).to_dict()
            tensorboard_logs = {**tensorboard_logs, **statistics}
        return {'validation_loss': loss,
                'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        genes, others, s, A, P, G, gene_names, other_names = batch
        seq, order = pack_sequences(genes, others)
        predA, theta, gap = self.aligner(seq, order)
        x, xlen, y, ylen = unpack_sequences(seq, order)
        loss = self.compute_loss(xlen, ylen, predA, A, P, G, theta)
        assert torch.isnan(loss).item() is False
        # Obtain alignment statistics + visualizations
        gen = self.aligner.traceback(seq, order)
        # TODO: compare the traceback and the forward
        statistics = self.validation_stats(
            x, y, xlen, ylen, gen, s, A, predA, theta, gap, batch_idx)
        assert len(statistics) > 0, (batch_idx, s)
        genes = list(map(
            lambda x: self.tokenizer.alphabet.decode(
                x.detach().cpu().numpy()).decode("utf-8"),
            genes))
        others = list(map(
            lambda x: self.tokenizer.alphabet.decode(
                x.detach().cpu().numpy()).decode("utf-8"),
            others))
        statistics = pd.DataFrame(
            statistics, columns=[
                'test_tp', 'test_fp', 'test_fn', 'test_perc_id',
                'test_ppv', 'test_fnr', 'test_fdr'
            ]
        )
        statistics['query_name'] = gene_names
        statistics['key_name'] = other_names
        return statistics

    def validation_epoch_end(self, outputs):
        loss_f = lambda x: x['validation_loss']
        losses = list(map(loss_f, outputs))
        loss = sum(losses) / len(losses)
        self.logger.experiment.add_scalar('val_loss', loss, self.global_step)
        # self.log('validation_loss') = loss

        # metrics = ['val_tp', 'val_fp', 'val_fn', 'val_perc_id',
        #            'val_ppv', 'val_fnr', 'val_fdr']
        # scores = []
        # for i, m in enumerate(metrics):
        #     loss_f = lambda x: x['log'][m]
        #     losses = list(map(loss_f, outputs))
        #     scalar = sum(losses) / len(losses)
        #     scores.append(scalar)
        #     self.logger.experiment.add_scalar(m, scalar, self.global_step)

        tensorboard_logs = dict(
            [('val_loss', loss)] # + list(zip(metrics, scores))
        )
        return {'val_loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # Freeze language model
        if self.hparams.finetune is False:
            for param in (self.aligner.lm.parameters()):
                param.requires_grad = False

        grad_params = list(filter(
            lambda p: p.requires_grad, self.aligner.parameters()))
        optimizer = torch.optim.AdamW(
            grad_params, lr=self.hparams.learning_rate)
        if self.hparams.scheduler == 'cosine_restarts':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=1, T_mult=2)
        elif self.hparams.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.epochs)
        elif self.hparams.scheduler == 'triangular':
            base_lr = 1e-8
            steps = int(np.log2(self.hparams.learning_rate / base_lr))
            steps = self.hparams.epochs // steps
            scheduler = CyclicLR(optimizer, base_lr,
                                 max_lr=self.hparams.learning_rate,
                                 step_size_up=steps,
                                 mode='triangular2',
                                 cycle_momentum=False)
        elif self.hparams.scheduler == 'steplr':
            m = 1e-6  # minimum learning rate
            steps = int(np.log2(self.hparams.learning_rate / m))
            steps = self.hparams.epochs // steps
            scheduler = StepLR(optimizer, step_size=steps, gamma=0.5)
        elif self.hparams.scheduler == 'none':
            return [optimizer]
        else:
            s = self.hparams.scheduler
            raise ValueError(f'`{s}` scheduler is not implemented.')
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
            '--pretrain-path', help='Path to pretrained protein language model',
            required=False, type=str, default='Rostlab/prot_t5_xl_uniref50')
        parser.add_argument(
            '--embedding-dim',
            help='Embedding dimension for aligner (default 1024).',
            required=False, type=int, default=1024)
        parser.add_argument(
            '--hidden-dim',
            help='Hidden dimension for intermediate layers (default 1024).',
            required=False, type=int, default=1024)
        parser.add_argument(
            '--layers', help='Number of RNN layers (default 2).',
            required=False, type=int, default=2)
        parser.add_argument(
            '--dropout', help='Dropout rate.',
            required=False, type=float, default=0.5)
        parser.add_argument(
            '--loss',
            help=('Loss function. Options include {sse, path, cross_entropy} '
                  '(default cross_entropy). '
                  'WARNING: this `path` loss is deprecated, '
                  'use at your own risk.'),
            default='cross_entropy', required=False, type=str)
        parser.add_argument(
            '--learning-rate', help='Learning rate',
            required=False, type=float, default=5e-5)
        parser.add_argument(
            '--batch-size', help='Training batch size',
            required=False, type=int, default=32)
        parser.add_argument(
            '--multitask', default=False, required=False, type=bool,
            help=(
                'Compute multitask loss between DP and matchings. '
                'WARNING: this option is deprecated, use at your own risk.'
            )
        )
        parser.add_argument(
            '--mode', default=False, required=False, type=str,
            help=(
                'Aligment mode. Choices include `needleman-wunch`, '
                '`smith-waterman`, or `gotoh`.'
            )
        )
        parser.add_argument(
            '--finetune',
            help=('Perform finetuning of language model.'),
            default=False, required=False, type=bool)
        parser.add_argument(
            '--mask-gaps',
            help=('Mask gaps from the loss calculation.'
                  'WARNING: this option is deprecated, use at your own risk.'),
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
            '--shuffle-validation',
            help='Whether or not to shuffle the validation data.',
            required=False, type=bool, default=False)
        parser.add_argument(
            '-o', '--output-directory',
            help='Output directory of model results', required=True)
        return parser
