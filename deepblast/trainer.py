import datetime
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from deepblast.alignment import NeedlemanWunschAligner
from deepblast.language_model import BiLM, pretrained_language_models
from deepblast.dataset.alphabet import UniprotTokenizer, Uniprot21
from deepblast.dataset import AlignmentDataset, collate
from deepblast.losses import SoftAlignmentLoss


class LightningAligner(pl.LightningModule):

    def __init__(self, args):
        path = pretrained_language_models['bilstm']
        self.embedding = BiLM()
        self.embedding.load_state_dict(torch.load(path))
        self.embedding.eval()
        self.tokenizer = UniprotTokenizer()
        self.hparams = args
        self.initialize_aligner()

    def initialize_aligner(self):
        n_alpha = len(self.tokenizer.alphabet)
        n_embed = self.args.embedding_dim
        n_input = self.args.rnn_input_dim
        n_units = self.args.rnn_dim
        if self.args.aligner_type == 'nw':
            self.aligner = NeedlemanWunschAligner(
                n_alpha, n_input, n_units, n_embed)
        else:
            raise NotImplemented(
                f'Aligner {self.args.aligner_type} not implemented.')

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
        pairs = pd.read_table(self.args.training_pairs, header=None)
        train_dataset = AlignmentDataset(pairs)
        train_dataloader = DataLoader(train_dataset, self.hparams.batch_size,
                                      shuffle=True, collate_fn=collate,
                                      num_workers=self.args.num_workers)
        return train_dataloader

    def valid_dataloader(self):
        pairs = pd.read_table(self.args.validation_pairs, header=None)
        valid_dataset = AlignmentDataset(pairs)
        valid_dataloader = DataLoader(valid_dataset, self.hparams.batch_size,
                                      shuffle=False, collate_fn=collate,
                                      num_workers=self.args.num_workers)
        return valid_dataloader

    def test_dataloader(self):
        pairs = pd.read_table(self.args.testing_pairs, header=None)
        test_dataset = AlignmentDataset(pairs)
        test_dataloader = DataLoader(test_dataset, self.hparams.batch_size,
                                     shuffle=False, collate_fn=collate,
                                     num_workers=self.args.num_workers)
        return test_dataloader

    def training_step(self, batch, batch_idx):
        x, y, s, A = batch
        self.aligner.train()
        predA = self.aligner(x, y)
        loss = SoftAlignmentLoss(A, predA)
        assert torch.isnan(loss).item) == False
        return {'training_loss': loss}

    def valid_step(self, batch, batch_idx):
        x, y, s, A = batch
        self.aligner.train()
        predA = self.aligner(x, y)
        loss = SoftAlignmentLoss(A, predA)
        assert torch.isnan(loss).item) == False
        # Measure the alignment accuracy
        return {'validation_loss': loss}

    def testing_step(self):
        # Measure the alignment accuracy
        pass

    def configure_optimizers(self):
        if not self.args.finetune:
            for p in self.model.lm.parameters():
                p.requires_grad = False
            grad_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            optimizer = torch.optim.RMSprop(
                grad_params, lr=self.args.learning_rate, weight_decay=self.args.reg_par)
        else:
            optimizer = torch.optim.RMSprop(
                [
                    {'params': self.model.lm.parameters(), 'lr': 5e-6},
                    {'params': self.model.aligner_fun.parameters(),
                     'lr': self.args.learning_rate,
                     'weight_decay': self.args.reg_par}
                ]
            )
        scheduler = ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument('--train-pairs', help='Training pairs file', required=True)
        parser.add_argument('--test-pairs', help='Testing pairs file', required=True)
        parser.add_argument('--valid-pairs', help='Validation pairs file', required=True)
        parser.add_argument('-m','--lm', help='Path to pretrained model',
                            required=False, default=None)
        parser.add_argument('-a','--aligner',
                            help='Aligner type. Choices include (mean, cca, ssa).',
                            required=False, type=str, default='mean')
        parser.add_argument('--embedding-dim', help='Embedding dimension (default 512).',
                            required=False, type=int, default=512)
        parser.add_argument('--rnn-input-dim', help='RNN input dimension (default 512).',
                            required=False, type=int, default=512)
        parser.add_argument('--rnn-dim', help='Number of hidden RNN units (default 512).',
                            required=False, type=int, default=512)
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
