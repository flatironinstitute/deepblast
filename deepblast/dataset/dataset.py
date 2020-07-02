import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
from deepblast.dataset.alphabet import UniprotTokenizer
from deepblast.constants import x, m, y


def state_f(z):
    if z[0] == '-':
        return x
    if z[1] == '-':
        return y
    else:
        return m


def tmstate_f(z):
    """ Parsing TM-specific state string. """
    if z == '1':
        return x
    if z == '2':
        return y
    else:
        return m


def clip_boundaries(X, Y, A):
    """ Remove xs and ys from ends. """
    first = A.index(m)
    last = len(A) - A[::-1].index(m)
    X, Y = states2alignment(A, X, Y)
    X_ = X[first:last].replace('-', '')
    Y_ = Y[first:last].replace('-', '')
    A_ = A[first:last]
    return X_, Y_, A_


def state_diff_f(X):
    """ Constructs a state transition element.

    Notes
    -----
    There is a bit of a paradox regarding beginning / ending gaps.
    To see this, try to derive an alignment matrix for the
    following alignments

    XXXMMMXXX
    MMYYXXMM

    It turns out it isn't possible to derive traversal rules
    that are consistent between these two alignments
    without explicitly handling start / end states as separate
    end states. The current workaround is to force the start / end
    states to be match states (similar to the needleman-wunsch algorithm).
    """
    a, b = X
    if a == x and b == x:
        # Transition XX, increase tape on X
        return (1, 0)
    if a == x and b == m:
        # Transition XM, increase tape on both X and Y
        return (1, 1)
    if a == m and b == m:
        # Transition MM, increase tape on both X and Y
        return (1, 1)
    if a == m and b == x:
        # Transition MX, increase tape on X
        return (1, 0)
    if a == m and b == y:
        # Transition MY, increase tape on y
        return (0, 1)
    if a == y and b == y:
        # Transition YY, increase tape on y
        return (0, 1)
    if a == y and b == m:
        # Transition YM, increase tape on both X and Y
        return (1, 1)
    if a == x and b == y:
        # Transition XY increase tape on y
        return (0, 1)
    if a == y and b == x:
        # Transition YX increase tape on x
        return (1, 0)
    else:
        raise ValueError(f'`Transition` ({a}, {b}) is not allowed.')


def states2edges(states):
    """ Converts state string to bipartite matching. """
    prev_s, next_s = states[:-1], states[1:]
    transitions = list(zip(prev_s, next_s))
    state_diffs = np.array(list(map(state_diff_f, transitions)))
    coords = np.cumsum(state_diffs, axis=0).tolist()
    coords = [(0, 0)] + list(map(tuple, coords))
    return coords


def states2matrix(states, sparse=False):
    """ Converts state string to alignment matrix.

    Parameters
    ----------
    states : list
       The state string
    N : int
       Length of sequence x.
    M : int
       Length of sequence y.
    """
    coords = states2edges(states)
    data = np.ones(len(coords))
    row, col = list(zip(*coords))
    row, col = np.array(row), np.array(col)
    N, M = max(row) + 1, max(col) + 1
    mat = coo_matrix((data, (row, col)), shape=(N, M))
    if sparse:
        return mat
    else:
        return mat.toarray()


def states2alignment(states, X, Y):
    """ Converts state string to gapped alignments """
    i, j = 0, 0
    res = []
    for k in range(len(states)):
        if states[k] == x:
            cx = X[i]
            cy = '-'
            i += 1
        elif states[k] == y:
            cx = '-'
            cy = Y[j]
            j += 1
        elif states[k] == m:
            cx = X[i]
            cy = Y[j]
            i += 1
            j += 1
        else:
            raise ValueError(f'{states[k]} is not recognized')
        res.append((cx, cy))

    aligned_x, aligned_y = zip(*res)
    return ''.join(aligned_x), ''.join(aligned_y)


def decode(codes, alphabet):
    """ Converts one-hot encodings to string

    Parameters
    ----------
    code : torch.Tensor
        One-hot encodings.
    alphabet : Alphabet
        Matches one-hot encodings to letters.

    Returns
    -------
    str
    """
    s = list(map(lambda x: alphabet[int(x)], codes))
    return ''.join(s)


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


class TMAlignDataset(AlignmentDataset):
    """ Dataset for training and testing.

    This is appropriate for the Malisam / Malidup datasets.
    """
    def __init__(self, path, tokenizer=UniprotTokenizer(),
                 tm_threshold=0.4, clip_ends=False, pad_ends=True):
        """ Read in pairs of proteins.


        This assumes that columns are labeled as
        | chain1_name | chain2_name | tmscore1 | tmscore2 | rmsd |
        | chain1 | chain2 | alignment |

        Parameters
        ----------
        patys: np.array of str
            Data path to aligned protein pairs.  This includes gaps
            and require that the proteins have the same length
        tokenizer: UniprotTokenizer
            Converts residues to one-hot encodings
        tm_threshold: float
            Minimum threshold to investigate alignments
        clip_ends: bools
            Removes gaps at the ends of the alignments.
            This will trim the sequences to force the tailing gaps
            to be removed. Default : False.

        Notes
        -----
        There are start/stop tokens that are incorporated into the
        alignment. The needleman-wunsch algorithm assumes this to be true.
        """
        self.tokenizer = tokenizer
        self.tm_threshold = tm_threshold
        self.pairs = pd.read_table(path, header=None)
        cols = [
            'chain1_name', 'chain2_name', 'tmscore1', 'tmscore2', 'rmsd',
            'chain1', 'chain2', 'alignment'
        ]
        self.pairs.columns = cols
        self.pairs['tm'] = np.maximum(
            self.pairs['tmscore1'], self.pairs['tmscore2'])
        idx = self.pairs['tm'] > self.tm_threshold
        self.pairs = self.pairs.loc[idx]
        self.clip_ends = clip_ends
        self.pad_ends = True

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
        gene = self.pairs.iloc[i]['chain1']
        pos = self.pairs.iloc[i]['chain2']
        states = self.pairs.iloc[i]['alignment']
        states = list(map(tmstate_f, states))
        # if self.clip_ends:  # TODO: this is broken. May want to remove.
        #     gene, pos, states = clip_boundaries(gene, pos, states)
        if self.pad_ends:
            states = [m] + states + [m]

        states = torch.Tensor(states).long()
        gene = self.tokenizer(str.encode(gene))
        pos = self.tokenizer(str.encode(pos))
        gene = torch.Tensor(gene).long()
        pos = torch.Tensor(pos).long()
        alignment_matrix = torch.from_numpy(
            states2matrix(states))
        return gene, pos, states, alignment_matrix


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
