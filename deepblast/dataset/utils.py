import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from deepblast.constants import x, m, y
from itertools import islice
from functools import reduce


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


def revstate_f(z):
    if z == x:
        return '1'
    if z == y:
        return '2'
    if z == m:
        return ':'


def clip_boundaries(X, Y, A, st):
    """ Remove xs and ys from ends. """
    if A[0] == m:
        first = 0
    else:
        first = A.index(m)

    if A[-1] == m:
        last = len(A)
    else:
        last = len(A) - A[::-1].index(m)
    X, Y = states2alignment(np.array(A), X, Y)
    X_ = X[first:last].replace('-', '')
    Y_ = Y[first:last].replace('-', '')
    A_ = A[first:last]
    st_ = st[first:last]
    return X_, Y_, A_, st_


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


def states2alignment(states: np.array, X: str, Y: str):
    """ Converts state string to gapped alignments """

    # Convert states to array if it is a string
    if isinstance(states, str):
        states = np.array(list(map(tmstate_f, list(states))))

    sx = np.sum(states == x) + np.sum(states == m)
    sy = np.sum(states == y) + np.sum(states == m)
    if sx != len(X):
        raise ValueError(
            f'The state string length {sx} does not match '
            f'the length of sequence {len(X)}.\n'
            f'SequenceX: {X}\nSequenceY: {Y}\nStates: {states}\n'
        )
    if sy != len(Y):
        raise ValueError(
            f'The state string length {sy} does not match '
            f'the length of sequence {len(X)}.\n'
            f'SequenceX: {X}\nSequenceY: {Y}\nStates: {states}\n'

        )

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


def pack_sequences(genes, others):
    x = genes + others
    lens = list(map(len, x))
    order = np.argsort(lens)[::-1].copy()
    y = [x[i] for i in order]
    packed = pack_sequence(y)
    return packed, order


def unpack_sequences(x, order):
    """ Unpack object into two sequences.

    Parameters
    ----------
    x : PackedSequence
        Packed sequence object containing 2 sequences.
    order : np.array
        The original order of the sequences.

    Returns
    -------
    x : torch.Tensor
        Tensor representation for first protein sequences.
    xlen : torch.Tensor
        Lengths of the first protein sequences.
    y : torch.Tensor
        Tensor representation for second protein sequences.
    ylen : torch.Tensor
        Lengths of the second protein sequences.
    """
    lookup = {order[i]: i for i in range(len(order))}
    seq, seqlen = pad_packed_sequence(x, batch_first=True)
    seq = [seq[lookup[i]] for i in range(len(order))]
    seqlen = [seqlen[lookup[i]] for i in range(len(order))]
    b = len(seqlen) // 2
    x, xlen = torch.stack(seq[:b]), torch.stack(seqlen[:b]).long()
    y, ylen = torch.stack(seq[b:]), torch.stack(seqlen[b:]).long()
    return x, xlen, y, ylen


def collate_f(batch):
    genes = [x[0] for x in batch]
    others = [x[1] for x in batch]
    states = [x[2] for x in batch]
    alignments = [x[3] for x in batch]
    paths = [x[4] for x in batch]
    masks = [x[5] for x in batch]
    max_x = max(map(len, genes))
    max_y = max(map(len, others))
    B = len(genes)
    dm = torch.zeros((B, max_x, max_y))
    p = torch.zeros((B, max_x, max_y))
    G = torch.zeros((B, max_x, max_y)).bool()
    G.requires_grad = False
    for b in range(B):
        n, m = len(genes[b]), len(others[b])
        dm[b, :n, :m] = alignments[b]
        p[b, :n, :m] = paths[b]
        G[b, :n, :m] = masks[b].bool()
    return genes, others, states, dm, p, G


def test_collate_f(batch):
    genes = [x[0] for x in batch]
    others = [x[1] for x in batch]
    states = [x[2] for x in batch]
    alignments = [x[3] for x in batch]
    paths = [x[4] for x in batch]
    masks = [x[5] for x in batch]
    gene_names = [x[6] for x in batch]
    other_names = [x[7] for x in batch]
    max_x = max(map(len, genes))
    max_y = max(map(len, others))
    B = len(genes)
    dm = torch.zeros((B, max_x, max_y))
    p = torch.zeros((B, max_x, max_y))
    G = torch.zeros((B, max_x, max_y)).bool()
    G.requires_grad = False
    for b in range(B):
        n, m = len(genes[b]), len(others[b])
        dm[b, :n, :m] = alignments[b]
        p[b, :n, :m] = paths[b]
        G[b, :n, :m] = masks[b].bool()
    return genes, others, states, dm, p, G, gene_names, other_names


def collate_fasta_f(batch):
    gene_ids = [x[0] for x in batch]
    other_ids = [x[1] for x in batch]
    genes = [x[2] for x in batch]
    others = [x[3] for x in batch]
    seqs, order = pack_sequences(genes, others)
    return gene_ids, other_ids, seqs, order


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

# Preprocessing functions
# def gap_mask(states: np.array):
#     """ Builds a mask for all gaps (0s are gaps, 1s are matches)
#
#     Parameters
#     ----------
#     states : np.array
#        List of alignment states
#
#     Returns
#     -------
#     mask : np.array
#        Masked array.
#
#     Notes
#     -----
#     Gaps and mismatches (denoted by `.`) are all masked here.
#     """x
#     i, j = 0, 0
#     res = []
#     coords = []
#     data = []
#     for k in range(len(states)):
#         # print(i, j, k, states[k])
#         if states[k] == '1':
#             coords.append((i, j))
#             data.append(0)
#             i += 1
#         elif states[k] == '2':
#             coords.append((i, j))
#             data.append(0)
#             j += 1
#         elif states[k] == ':':
#             coords.append((i, j))
#             data.append(1)
#             i += 1
#             j += 1
#         elif states[k] == '.':
#             coords.append((i, j))
#             data.append(0)
#             i += 1
#             j += 1
#         else:
#             raise ValueError(f'{states[k]} is not recognized')
#     rows, cols = zip(*coords)
#     rows = np.array(rows)
#     cols = np.array(cols)
#     data = np.array(data)
#     mask = coo_matrix((data, (rows, cols))).todense()
#     return mask


def gap_mask(states: str, sparse=False):
    st = np.array(list(map(tmstate_f, list(states))))
    coords = states2edges(st)
    data = np.ones(len(coords))
    row, col = list(zip(*coords))
    row, col = np.array(row), np.array(col)
    N, M = max(row) + 1, max(col) + 1
    idx = np.array(list(states)) == ':'
    idx[0] = 1
    data = data[idx]
    row = row[idx]
    col = col[idx]
    mat = coo_matrix((data, (row, col)), shape=(N, M))
    if sparse:
        return mat
    else:
        return mat.toarray().astype(np.bool)


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def replace_orphan(w, s=5):
    i = len(w) // 2
    # identify orphans and replace with gaps
    sw = ''.join(w)
    if ((w[i] == ':') and ((('1' * s) in sw[:i] and ('1' * s) in sw[i:])
                           or (('2' * s) in sw[:i] and ('2' * s) in sw[i:]))):
        return ['1', '2']
    else:
        return [w[i]]


def remove_orphans(states, threshold: int = 11):
    """ Removes singletons and doubletons that are orphaned.

    A match is considered orphaned if it exceeds the `threshold` gap.

    Parameters
    ----------
    states : np.array
       List of alignment states
    threshold : int
       Number of consecutive gaps surrounding a matched required for it
       to be considered an orphan.

    Returns
    -------
    new_states : np.array
       States string with orphans removed.

    Notes
    -----
    The threshold *must* be an odd number. This determines the window size.
    """
    wins = list(window(states, threshold))
    rwins = list(map(lambda x: replace_orphan(x, threshold // 2), list(wins)))
    new_states = list(reduce(lambda x, y: x + y, rwins))
    new_states += list(states[:threshold // 2])
    new_states += list(states[-threshold // 2 + 1:])
    return ''.join(new_states)
