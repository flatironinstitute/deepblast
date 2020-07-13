import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
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
    genes : list of Tensor
        List of proteins
    others : list of Tensor
        List of proteins
    states : list of Tensor
        List of alignment state strings
    dm : torch.Tensor
        B x N x M dimension matrix with padding.
    """
    s = list(map(lambda x: alphabet[int(x)], codes))
    return ''.join(s)


def collate_f(batch):
    genes = [x[0] for x in batch]
    others = [x[1] for x in batch]
    states = [x[2] for x in batch]
    alignments = [x[3] for x in batch]
    paths = [x[4] for x in batch]
    max_x = max(map(len, genes))
    max_y = max(map(len, others))
    B = len(genes)
    dm = torch.zeros((B, max_x, max_y))
    p = torch.zeros((B, max_x, max_y))
    # gene_codes = torch.zeros((B, max_x), dtype=torch.long)
    # other_codes = torch.zeros((B, max_y), dtype=torch.long)
    for b in range(B):
        n, m = len(genes[b]), len(others[b])
        dm[b, :n, :m] = alignments[b]
        p[b, :n, :m] = paths[b]
        # gene_codes[b, :n] = genes[b]
        # other_codes[b, :m] = others[b]
    return genes, others, states, dm, p


def path_distance_matrix(pi):
    """ Builds a min path distance matrix.

    This will be passed into the SoftPathLoss function.
    For each cell, it will compute the distance between
    coordinates in the cell and the nearest cell located in the path.

    Parameters
    ----------
    pi : list of tuple
       Coordinates of the ground truth alignment

    Returns
    -------
    Pdist : np.array
       Matrix of distances to path.
    """
    pi = np.array(pi)
    model = cKDTree(pi)
    xs = np.arange(pi[:, 0].max() + 1)
    ys = np.arange(pi[:, 1].max() + 1)
    coords = np.dstack(np.meshgrid(xs, ys)).reshape(-1, 2)
    d, i = model.query(coords)
    Pdist = np.array(coo_matrix((d, (coords[:, 0], coords[:, 1]))).todense())
    return Pdist


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
                 tm_threshold=0.4, max_len=1024, pad_ends=True,
                 construct_paths=False):
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
        max_len : float
            Maximum sequence length to be aligned
        pad_ends : bool
            Specifies if the ends of the sequences should be padded or not.
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
        path_matrix : torch.Tensor
           Pairwise path distances, where the smallest distance
           to the path is computed for every element in the matrix.
        """
        gene = self.pairs.iloc[i]['chain1']
        pos = self.pairs.iloc[i]['chain2']
        states = self.pairs.iloc[i]['alignment']
        states = list(map(tmstate_f, states))
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

        return gene, pos, states, alignment_matrix, path_matrix


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
