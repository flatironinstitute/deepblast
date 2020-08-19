import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import Dataset
from deepblast.dataset.alphabet import UniprotTokenizer
from deepblast.constants import m
from deepblast.dataset.utils import (
    state_f, tmstate_f,
    clip_boundaries, states2matrix, states2edges,
    path_distance_matrix, gap_mask
)


def reshape(x, N, M):
    if x.shape != (N, M) and x.shape != (M, N):
        raise ValueError(f'The shape of `x` {x.shape} '
                         f'does not agree with ({N}, {M})')
    if tuple(x.shape) != (N, M):
        return x.t()
    else:
        return x


class AlignmentDataset(Dataset):
    def __init__(self, pairs, tokenizer=UniprotTokenizer()):
        self.tokenizer = tokenizer
        self.pairs = pairs

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = len(self.pairs)

        if worker_info is None:  # single-process data loading
            for i in range(end):
                yield self.__getitem__(i)
        else:
            worker_id = worker_info.id
            w = float(worker_info.num_workers)
            t = (end - start)
            w = float(worker_info.num_workers)
            per_worker = int(math.ceil(t / w))
            worker_id = worker_info.id
            iter_start = start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, end)
            for i in range(iter_start, iter_end):
                yield self.__getitem__(i)


class FastaDataset(AlignmentDataset):
    """ Dataset for searching. """
    def __init__(self, query_path, db_path, tokenizer=UniprotTokenizer()):
        pass


class TMAlignDataset(AlignmentDataset):
    """ Dataset for training and testing.

    This is appropriate for the Malisam / Malidup datasets.
    """
    def __init__(self, path, tokenizer=UniprotTokenizer(),
                 tm_threshold=0.4, max_len=1024, pad_ends=False,
                 clip_ends=True, construct_paths=False):
        """ Read in pairs of proteins.


        This assumes that columns are labeled as
        | chain1_name | chain2_name | tmscore1 | tmscore2 | rmsd |
        | chain1 | chain2 | alignment |

        Parameters
        ----------
        path: path
            Data path to aligned protein pairs.  This includes gaps
            and require that the proteins have the same length
        tokenizer: UniprotTokenizer
            Converts residues to one-hot encodings
        tm_threshold: float
            Minimum threshold to investigate alignments
        max_len : float
            Maximum sequence length to be aligned
        pad_ends : bool
            Specifies if the ends of the sequences should be padded or not.
        clip_ends : bool
            Specifies if the ends of the alignments should be clipped or not.
        construct_paths : bool
            Specifies if path distances should be calculated.

        Notes
        -----
        There are start/stop tokens that are incorporated into the
        alignment. The needleman-wunsch algorithm assumes this to be true.
        """
        self.tokenizer = tokenizer
        self.tm_threshold = tm_threshold
        self.max_len = max_len
        self.pairs = pd.read_table(path, header=None)
        self.construct_paths = construct_paths
        cols = [
            'chain1_name', 'chain2_name', 'tmscore1', 'tmscore2', 'rmsd',
            'chain1', 'chain2', 'alignment'
        ]
        self.pairs.columns = cols
        self.pairs['tm'] = np.maximum(
            self.pairs['tmscore1'], self.pairs['tmscore2'])
        self.pairs['length'] = self.pairs.apply(
            lambda x: max(len(x['chain1']), len(x['chain2'])), axis=1)
        idx = np.logical_and(self.pairs['tm'] > self.tm_threshold,
                             self.pairs['length'] < self.max_len)
        self.pairs = self.pairs.loc[idx]
        # TODO: pad_ends needs to be documented properly
        self.pad_ends = pad_ends
        self.clip_ends = clip_ends

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, i):
        """ Gets alignment pair.

        Parameters
        ----------
        i : int
           Index of item

        Returns
        -------
        gene : torch.Tensor
           Encoded representation of protein of interest
        pos : torch.Tensor
           Encoded representation of protein that aligns with `gene`.
        states : torch.Tensor
           Alignment string
        alignment_matrix : torch.Tensor
           Ground truth alignment matrix
        path_matrix : torch.Tensor
           Pairwise path distances, where the smallest distance
           to the path is computed for every element in the matrix.
        """
        gene = self.pairs.iloc[i]['chain1']
        pos = self.pairs.iloc[i]['chain2']
        states = self.pairs.iloc[i]['alignment']

        gene_mask, pos_mask = gap_mask(states)
        states = list(map(tmstate_f, states))
        if self.clip_ends:
            gene, pos, states = clip_boundaries(gene, pos, states)

        if self.pad_ends:
            states = [m] + states + [m]

        states = torch.Tensor(states).long()
        gene = self.tokenizer(str.encode(gene))
        pos = self.tokenizer(str.encode(pos))
        gene = torch.Tensor(gene).long()
        pos = torch.Tensor(pos).long()
        alignment_matrix = torch.from_numpy(
            states2matrix(states))
        path_matrix = torch.empty(*alignment_matrix.shape)
        if self.construct_paths:
            pi = states2edges(states)
            path_matrix = torch.from_numpy(path_distance_matrix(pi))

        if tuple(path_matrix.shape) != (len(gene), len(pos)):
            path_matrix = path_matrix.t()
        if tuple(alignment_matrix.shape) != (len(gene), len(pos)):
            alignment_matrix = alignment_matrix.t()

        gene_mask = torch.Tensor(gene_mask)
        pos_mask = torch.Tensor(pos_mask)

        return (gene, pos, states, alignment_matrix, path_matrix,
                gene_mask, pos_mask)


class MaliAlignmentDataset(AlignmentDataset):
    """ Dataset for training and testing Mali datasets

    This is appropriate for the Malisam / Malidup datasets.
    """
    def __init__(self, pairs, tokenizer=UniprotTokenizer()):
        """ Read in pairs of proteins

        Parameters
        ----------
        pairs: np.array of str
            Pairs of proteins that are aligned.  This includes gaps
            and require that the proteins have the same length
        """
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, i):
        """ Gets alignment pair.

        Parameters
        ----------
        i : int
           Index of item

        Returns
        -------
        gene : torch.Tensor
           Encoded representation of protein of interest
        pos : torch.Tensor
           Encoded representation of protein that aligns with `gene`.
        states : torch.Tensor
           Alignment string
        alignment_matrix : torch.Tensor
           Ground truth alignment matrix
        """
        gene = self.pairs.loc[i, 0]
        pos = self.pairs.loc[i, 1]
        assert len(gene) == len(pos)
        alnstr = list(zip(list(gene), list(pos)))
        states = torch.Tensor(list(map(state_f, alnstr)))
        gene = self.tokenizer(str.encode(gene.replace('-', '')))
        pos = self.tokenizer(str.encode(pos.replace('-', '')))
        gene = torch.Tensor(gene).long()
        pos = torch.Tensor(pos).long()
        alignment_matrix = torch.from_numpy(states2matrix(states))
        return gene, pos, states, alignment_matrix
