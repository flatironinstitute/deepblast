# Installation

DeepBLAST can be installed from pip via

```
pip install deepblast
```

To install from the development branch run

```
pip install git+https://github.com/flatironinstitute/deepblast.git
```

# Downloading pretrained models and data

The pretrained DeepBLAST model can be downloaded [here](https://users.flatironinstitute.org/mortonjt/public_www/deepblast-public-data/checkpoints/epoch=9.pt).

The TM-align structural alignments used to pretrain DeepBLAST can be found [here](https://users.flatironinstitute.org/mortonjt/public_www/deepblast-public-data/tmalign.tar.gz)


See the [Malisam](http://prodata.swmed.edu/malisam/) and [Malidup](http://prodata.swmed.edu/malidup/) websites to download their datasets.



# Getting started

We have 3 command line scripts available, namely `deepblast-train`, `deepblast-eval` and `deepblast-search`.

## Pretraining

`deepblast-train` takes in as input a tab-delimited format of with columns
`query_seq_id | key_seq_id | tm_score1 | tm_score2 | rmsd | sequence1 | sequence2 | alignment_string`
See an example [here](https://raw.githubusercontent.com/flatironinstitute/deepblast/master/data/tm_align_output_10k.tab) of what this looks like. At this moment, we only support parsing the output of TM-align. The parsing script can be found under

`deepblast/dataset/parse_tm_align.py [fname] [output_table]`

Once the data is configured and split appropriately, `deepblast-train` can be run.
The command-line options are given below (see `deepblast-train --help` for more details).

```
usage: deepblast-train [-h] [--gpus GPUS] [--grad-accum GRAD_ACCUM] [--grad-clip GRAD_CLIP] [--nodes NODES] [--num-workers NUM_WORKERS] [--precision PRECISION] [--backend BACKEND]
                       [--load-from-checkpoint LOAD_FROM_CHECKPOINT] --train-pairs TRAIN_PAIRS --test-pairs TEST_PAIRS --valid-pairs VALID_PAIRS [--embedding-dim EMBEDDING_DIM]
                       [--rnn-input-dim RNN_INPUT_DIM] [--rnn-dim RNN_DIM] [--layers LAYERS] [--loss LOSS] [--learning-rate LEARNING_RATE] [--batch-size BATCH_SIZE]
                       [--multitask MULTITASK] [--finetune FINETUNE] [--mask-gaps MASK_GAPS] [--scheduler SCHEDULER] [--epochs EPOCHS]
                       [--visualization-fraction VISUALIZATION_FRACTION] -o OUTPUT_DIRECTORY
```

## Evaluation

This will evaluate how much the deepblast predictions agree with the structural alignments.
The `deepblast-train` command will automatically evaluate the heldout test set if it completes.
However, a separate `deepblast-evaluate` command is available in case the pretraining was interrupted.  The commandline options are given below (see `deepblast-evaluate --help` for more details)

```
usage: deepblast-evaluate [-h] [--gpus GPUS] [--num-workers NUM_WORKERS] [--nodes NODES] [--load-from-checkpoint LOAD_FROM_CHECKPOINT] [--precision PRECISION] [--backend BACKEND]
                          --train-pairs TRAIN_PAIRS --test-pairs TEST_PAIRS --valid-pairs VALID_PAIRS [--embedding-dim EMBEDDING_DIM] [--rnn-input-dim RNN_INPUT_DIM]
                          [--rnn-dim RNN_DIM] [--layers LAYERS] [--loss LOSS] [--learning-rate LEARNING_RATE] [--batch-size BATCH_SIZE] [--multitask MULTITASK]
                          [--finetune FINETUNE] [--mask-gaps MASK_GAPS] [--scheduler SCHEDULER] [--epochs EPOCHS] [--visualization-fraction VISUALIZATION_FRACTION] -o
                          OUTPUT_DIRECTORY
```

## Search

We have enabled a simple fasta search that will enable structural similarity to be evaluated across fasta files.  This will perform a GPU-accelerated Needleman-Wunsch to evaluate all pairwise alignments.  At this moment, we only output tab-delimited file of the alignment scores.  The commandline options are given below (see `deepblast-search --help` for more details)

```
usage: deepblast-search [-h] --query-fasta QUERY_FASTA --db-fasta DB_FASTA --load-from-checkpoint LOAD_FROM_CHECKPOINT --output-file OUTPUT_FILE [--gpu GPU]
                        [--num-workers NUM_WORKERS] [--batch-size BATCH_SIZE]
```

## Loading the models

```python

# Load checkpoint
from deepblast.trainer import LightningAligner
from deepblast.dataset.utils import pack_sequences
from deepblast.dataset.utils import states2alignment
import matplotlib.pyplot as plt

# Load the pretrained model
model = LightningAligner.load_from_checkpoint(your_model_path)

# Load on GPU (if you want)
model = model.cuda()

# Obtain hard alignment from the raw sequences
x = 'IGKEEIQQRLAQFVDHWKELKQLAAARGQRLEESLEYQQFVANVEEEEAWINEKMTLVASED'
y = 'QQNKELNFKLREKQNEIFELKKIAETLRSKLEKYVDITKKLEDQNLNLQIKISDLEKKLSDA'
pred_alignment = model.align(x, y)
x_aligned, y_aligned = states2alignment(states, x, y)
print(x_aligned)
print(states)
print(y_aligned)

x = str.encode('IGKEEIQQRLAQFVDHWKELKQLAAARGQRLEESLEYQQFVANVEEEEAWINEKMTLVASED')
y = str.encode('QQNKELNFKLREKQNEIFELKKIAETLRSKLEKYVDITKKLEDQNLNLQIKISDLEKKLSDA')
x_ = torch.Tensor(model.tokenizer(x)).long()
y_ = torch.Tensor(model.tokenizer(y)).long()

# Pack sequences for easier parallelization
seq, order = pack_sequences([x_], [y_])
seq = seq.cuda()

# Predict expected alignment
A, match_scores, gap_scores = model.forward(seq, order)

# Display the expected alignment
fig, ax = plt.subplots(1, 3, figsize=(9, 3))
sns.heatmap(aln.cpu().detach().numpy().squeeze(), ax=ax[0], cbar=False,  cmap='viridis')
sns.heatmap(theta.cpu().detach().numpy().squeeze(), ax=ax[1],  cmap='viridis')
sns.heatmap(g.cpu().detach().numpy().squeeze(), ax=ax[2],  cmap='viridis')
ax[0].set_title('Predicted Alignment')
ax[1].set_title('Match scores ($\mu$)')
ax[2].set_title('Gap scores ($g$)')
plt.tight_layout()
plt.show()
```

The output will look like
```
IGKEEIQQRLAQFVDHWKELKQLAAARGQRLEESLEYQQFVANVEEEEAWINEKMTLVASED
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
QQNKELNFKLREKQNEIFELKKIAETLRSKLEKYVDITKKLEDQNLNLQIKISDLEKKLSDA
```

![](https://raw.githubusercontent.com/flatironinstitute/deepblast/master/imgs/example-alignment.png "example alignment")
