import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from scipy.sparse import coo_matrix
from deepblast.dataset.alphabet import UniprotTokenizer


def state_f(z):
    x, m, y = 0, 1, 2 # state numberings
    if z[0] == '-':
        return x
    if z[1] == '-':
        return y
    else:
        return m

def states2matrix(states, N, M):
    """ Converts state string to alignment matrix. """
    x, m, y = 0, 1, 2 # state numberings
    # Encode as sparse matrix
    i, j = 0, 0
    coords = [(i, j)]
    for st in states:
        if st == x:
            j += 1
        elif st == y:
            i += 1
        else:
            i += 1
            j += 1
        coords.append((i, j))
    data = np.ones(len(coords))
    row, col = list(zip(*coords))
    row, col = np.array(row), np.array(col)
    mat = coo_matrix((data, (row, col)), shape=(N, M)).toarray()
    return mat


class AlignmentDataset(Dataset):
    """ Dataset for training and testing. """
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
        N, M = len(gene), len(pos)
        alignment_matrix = torch.from_numpy(states2matrix(states, N, M))
        return gene, pos, states, alignment_matrix

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
