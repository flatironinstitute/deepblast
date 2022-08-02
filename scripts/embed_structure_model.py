import json
import inspect
from functools import partial
from dataclasses import dataclass, asdict

import torch
from torch import nn
import pytorch_lightning as pl


@dataclass
class Config:
    def isolate(self, config):
        specifics = inspect.signature(config).parameters
        my_specifics = {k: v for k, v in asdict(self).items() if k in specifics}
        return config(**my_specifics)

    def to_json(self, filename):
        config = json.dumps(asdict(self), indent=2)
        with open(filename, 'w') as f:
            f.write(config)
    
    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as f:
            js = json.loads(f.read())
        config = cls(**js)
        return config


@dataclass
class trans_basic_block_Config(Config):
    d_model: int = 1024
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 2048
    out_dim: int = 512
    dropout: float = 0.1
    activation: str = 'relu'
    # data params
    lr0: float = 0.0001
    warmup_steps: int = 300

    def build(self):
        return trans_basic_block(self)


class trans_basic_block(pl.LightningModule):
    """
    TransformerEncoderLayer with preset parameters followed by global pooling and dropout
    """
    def __init__(self, config: trans_basic_block_Config):
        super().__init__()
        self.config = config

        # build encoder
        encoder_args = {k: v for k, v in asdict(config).items() if k in inspect.signature(nn.TransformerEncoderLayer).parameters} 
        num_layers = config.num_layers

        encoder_layer = nn.TransformerEncoderLayer(batch_first=True, **encoder_args)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(self.config.dropout)
        self.mlp = nn.Linear(self.config.d_model, self.config.out_dim)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, x, src_mask, src_key_padding_mask):
        x = self.encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        lens = torch.logical_not(src_key_padding_mask).sum(dim=1).float()
        x = x.sum(dim=1) / lens.unsqueeze(1)
        x = self.dropout(x)
        x = self.mlp(x)
        return x

    
    def distance_loss_euclidean(self, output_seq1, output_seq2, tm_score):
        pdist_seq = nn.PairwiseDistance(p=2)
        dist_seq = pdist_seq(output_seq1, output_seq2)
        dist_tm = torch.cdist(dist_seq.unsqueeze(0), tm_score.float().unsqueeze(0), p=2)
        return dist_tm

    def distance_loss_sigmoid(self, output_seq1, output_seq2, tm_score):
        dist_seq = output_seq1 - output_seq2
        dist_seq = torch.sigmoid(dist_seq).mean(1)
        dist_tm = torch.cdist(dist_seq.unsqueeze(0), tm_score.float().unsqueeze(0), p=2)
        return dist_tm

    def distance_loss(self, output_seq1, output_seq2, tm_score):
        dist_seq = self.cos(output_seq1, output_seq2)  
        dist_tm = self.l1_loss(dist_seq.unsqueeze(0), tm_score.float().unsqueeze(0))
        return dist_tm

    def training_step(self, train_batch, batch_idx):
        sequence_1, sequence_2, pad_mask_1, pad_mask_2, tm_score = train_batch
        out_seq1 = self.forward(sequence_1, src_mask=None, src_key_padding_mask=pad_mask_1)
        out_seq2 = self.forward(sequence_2, src_mask=None, src_key_padding_mask=pad_mask_2)
        loss = self.distance_loss(out_seq1, out_seq2, tm_score)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        sequence_1, sequence_2, pad_mask_1, pad_mask_2, tm_score = val_batch
        out_seq1 = self.forward(sequence_1, src_mask=None, src_key_padding_mask=pad_mask_1)
        out_seq2 = self.forward(sequence_2, src_mask=None, src_key_padding_mask=pad_mask_2)
        loss = self.distance_loss(out_seq1, out_seq2, tm_score)
        self.log('val_loss', loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.lr0)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [lr_scheduler]
