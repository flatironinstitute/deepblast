import datetime
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from deepblast.alignment import NeedlemanWunschAligner
from deepblast.dataset.alphabet import UniprotTokenizer
from deepblast.dataset import TMAlignDataset
from deepblast.losses import SoftAlignmentLoss


class LightningAligner(pl.LightningModule):

    def __init__(self, args):
        super(LightningAligner, self).__init__()
        self.tokenizer = UniprotTokenizer()
        self.hparams = args
        self.initialize_aligner()
        self.loss_func = SoftAlignmentLoss()

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
        print(self.hparams.train_pairs)
        train_dataset = TMAlignDataset(
            self.hparams.train_pairs, clip_ends=self.hparams.clip_ends)
        train_dataloader = DataLoader(
            train_dataset, self.hparams.batch_size,
            shuffle=True, num_workers=self.hparams.num_workers)
        return train_dataloader

    def val_dataloader(self):
        valid_dataset = TMAlignDataset(self.hparams.valid_pairs)
        valid_dataloader = DataLoader(
            valid_dataset, self.hparams.batch_size,
            shuffle=False, num_workers=self.hparams.num_workers)
        return valid_dataloader

    def test_dataloader(self):
        test_dataset = TMAlignDataset(self.hparams.test_pairs)
        test_dataloader = DataLoader(
            test_dataset, self.hparams.batch_size, shuffle=False, 
            num_workers=self.hparams.num_workers)
        return test_dataloader

    def training_step(self, batch, batch_idx):
        x, y, s, A = batch
        self.aligner.train()
        predA = self.aligner(x, y)
        loss = self.loss_func(A, predA)
        assert torch.isnan(loss).item() is False
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y, s, A = batch
        self.aligner.train()
        predA = self.aligner(x, y)
        loss = SoftAlignmentLoss(A, predA)
        assert torch.isnan(loss).item() is False
        tensorboard_logs = {'valid_loss': loss}
        # TODO: Measure the alignment accuracy
        return {'validation_loss': loss, 
                'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y, s, A = batch
        self.aligner.train()
        predA = self.aligner(x, y)
        loss = SoftAlignmentLoss(A, predA)
        assert torch.isnan(loss).item() is False
        # TODO: Measure the alignment accuracy
        return {'test_loss': loss}

    def validation_epoch_end(self, outputs):
        loss_f = lambda x: x['validation_loss']
        losses = list(map(loss_f, outputs))
        val_loss = sum(losses) / len(losses)
        results = {'validation_loss' : val_loss}
        return results      

    def configure_optimizers(self):
        for p in self.aligner.lm.parameters():
            p.requires_grad = False
        grad_params = list(filter(
            lambda p: p.requires_grad, self.aligner.parameters()))    
        optimizer = torch.optim.Adam(
            self.aligner.parameters(), lr=self.hparams.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.epochs)
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
            '--learning-rate', help='Learning rate',
            required=False, type=float, default=5e-5)
        parser.add_argument(
            '--batch-size', help='Training batch size',
            required=False, type=int, default=32)
        parser.add_argument(
            '--finetune', help='Perform finetuning (does not work with mean)',
            default=False, required=False, type=bool)
        parser.add_argument(
            '--clip-ends',
            help=('Specifies if training start/end gaps should be removed. '
                  'This will speed up runtime.'),
            default=False, required=False, type=bool)
        parser.add_argument(
            '--epochs', help='Training batch size',
            required=False, type=int, default=10)
        parser.add_argument(
            '-o', '--output-directory',
            help='Output directory of model results', required=True)
        return parser
