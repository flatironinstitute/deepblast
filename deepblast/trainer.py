import datetime
import argparse
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from deepblast.alignment import NeedlemanWunschAligner
from deepblast.language_model import BiLM, pretrained_language_models
from deepblast.dataset.alphabet import UniprotTokenizer, Uniprot21
from deepblast.dataset import AlignmentDataset
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
            raise NotImplemented(
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
        pairs = pd.read_table(self.hparams.train_pairs, header=None)
        train_dataset = AlignmentDataset(pairs)
        train_dataloader = DataLoader(
            train_dataset, self.hparams.batch_size,
            shuffle=True, num_workers=self.hparams.num_workers)
        return train_dataloader

    def valid_dataloader(self):
        pairs = pd.read_table(self.hparams.valid_pairs, header=None)
        valid_dataset = AlignmentDataset(pairs)
        valid_dataloader = DataLoader(
            valid_dataset, self.hparams.batch_size,
            shuffle=False, num_workers=self.hparams.num_workers)
        return valid_dataloader

    def test_dataloader(self):
        pairs = pd.read_table(self.hparams.testing_pairs, header=None)
        test_dataset = AlignmentDataset(pairs)
        test_dataloader = DataLoader(test_dataset, self.hparams.batch_size,
                                     shuffle=False, collate_fn=collate,
                                     num_workers=self.hparams.num_workers)
        return test_dataloader

    def training_step(self, batch, batch_idx):
        x, y, s, A = batch
        self.aligner.train()
        predA = self.aligner(x, y)
        loss = self.loss_func(A, predA)
        assert torch.isnan(loss).item() == False
        return {'loss': loss}

    def valid_step(self, batch, batch_idx):
        x, y, s, A = batch
        self.aligner.train()
        predA = self.aligner(x, y)
        loss = SoftAlignmentLoss(A, predA)
        assert torch.isnan(loss).item() == False
        # TODO: Measure the alignment accuracy
        return {'validation_loss': loss}

    def test_step(self):
        # TODO: Measure the alignment accuracy
        pass

    def configure_optimizers(self):
        for p in self.aligner.lm.parameters():
            p.requires_grad = False
        grad_params = list(filter(
            lambda p: p.requires_grad, self.aligner.parameters()))
        optimizer = torch.optim.Adam(
            grad_params, lr=self.hparams.learning_rate)
        scheduler = StepLR(optimizer, step_size=10)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--train-pairs', help='Training pairs file', required=True)
        parser.add_argument('--test-pairs', help='Testing pairs file', required=True)
        parser.add_argument('--valid-pairs', help='Validation pairs file', required=True)
        parser.add_argument('-a','--aligner',
                            help='Aligner type. Choices include (nw, hmm).',
                            required=False, type=str, default='nw')
        parser.add_argument('--embedding-dim', help='Embedding dimension (default 512).',
                            required=False, type=int, default=512)
        parser.add_argument('--rnn-input-dim', help='RNN input dimension (default 512).',
                            required=False, type=int, default=512)
        parser.add_argument('--rnn-dim', help='Number of hidden RNN units (default 512).',
                            required=False, type=int, default=512)
        parser.add_argument('--layers', help='Number of RNN layers (default 2).',
                            required=False, type=int, default=2)
        parser.add_argument('--learning-rate', help='Learning rate',
                            required=False, type=float, default=5e-5)
        parser.add_argument('--batch-size', help='Training batch size',
                            required=False, type=int, default=32)
        parser.add_argument('--finetune', help='Perform finetuning (does not work with mean)',
                            default=False, required=False, type=bool)
        parser.add_argument('--epochs', help='Training batch size',
                            required=False, type=int, default=10)
        parser.add_argument('-o','--output-directory', help='Output directory of model results',
                            required=True)
        return parser
