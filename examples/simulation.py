import os
from deepblast.sim import hmm_alignments
import argparse
import numpy as np
from pytorch_lightning import Trainer
from transformers import T5EncoderModel, T5Tokenizer
from deepblast.trainer import DeepBLAST


# create simulation dataset
hmm = '../data/zf-C2H2.hmm'
n_alignments = 100
np.random.seed(0)
align_df = hmm_alignments(n=40, seed=0, n_alignments=n_alignments, hmmfile=hmm)

cols = [
    'chain1_name', 'chain2_name', 'tmscore1', 'tmscore2', 'rmsd',
    'chain1', 'chain2', 'alignment'
]
align_df.columns = cols

# split into train/test/validation dataset
parts = n_alignments // 10
train_df = align_df.iloc[:parts * 8]
test_df = align_df.iloc[parts * 8:parts * 9]
valid_df = align_df.iloc[parts * 9:]

# save the files to disk.
if not os.path.exists('data'):
    os.mkdir('data')

train_df.to_csv('data/train.txt', sep='\t', index=None, header=None)
test_df.to_csv('data/test.txt', sep='\t', index=None, header=None)
valid_df.to_csv('data/valid.txt', sep='\t', index=None, header=None)

output_dir = 'simulation_results'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Load the protrans model
tokenizer = T5Tokenizer.from_pretrained(
    "Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
lm = T5EncoderModel.from_pretrained(
    "Rostlab/prot_t5_xl_uniref50")


# Create the deepblast model
model = DeepBLAST(
    train_pairs=f'{os.getcwd()}/data/train.txt',
    test_pairs=f'{os.getcwd()}/data/test.txt',
    valid_pairs=f'{os.getcwd()}/data/valid.txt',
    output_directory=output_dir,
    hidden_dim=1024,
    embedding_dim=1024,
    batch_size=10,
    num_workers=10,
    layers=1,
    learning_rate=5e-5,
    loss='cross_entropy',
    lm=lm,
    tokenizer=tokenizer
)

# Fit the DeepBLAST model
trainer = Trainer(
    max_epochs=10,
    limit_train_batches=10,  # short run, we'll only train 10 batches / epoch
    limit_val_batches=10,    # short run, ...
    gpus=1,
    check_val_every_n_epoch=1,
    # profiler=profiler,
    fast_dev_run=True,
    # auto_scale_batch_size='power'
)

trainer.fit(model)
