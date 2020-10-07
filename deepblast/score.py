import numpy as np
import matplotlib.pyplot as plt
from deepblast.dataset.utils import states2alignment, tmstate_f, states2edges


def roc_edges(true_edges, pred_edges):
    truth = set(true_edges)
    pred = set(pred_edges)
    tp = len(truth & pred)
    fp = len(pred - truth)
    fn = len(truth - pred)
    perc_id = tp / len(true_edges)
    ppv = tp / (tp + fp)
    fnr = fn / (fn + tp)
    fdr = fp / (fp + tp)
    return tp, fp, fn, perc_id, ppv, fnr, fdr


def roc_edges_kernel_identity(true_edges, pred_edges, kernel_width):
    pe_ = pred_edges
    pe = np.array(pred_edges)
    for k in range(kernel_width):
        pred_edges_k_pos = pe + k
        pred_edges_k_neg = pe - k
        pe_ += list(map(tuple, pred_edges_k_pos))
        pe_ += list(map(tuple, pred_edges_k_neg))

    truth = set(true_edges)
    pred = set(pe_)
    tp = len(truth & pred)
    perc_id = tp / len(true_edges)
    return perc_id


def alignment_score_kernel(true_states: str, pred_states: str,
                           kernel_widths: list,
                           query_offset: int = 0, hit_offset: int = 0):
    """
    Computes ROC statistics on alignment

    Parameters
    ----------
    true_states : str
        Ground truth state string
    pred_states : str
        Predicted state string
    """

    pred_states = list(map(tmstate_f, pred_states))
    true_states = list(map(tmstate_f, true_states))
    pred_edges = states2edges(pred_states)
    true_edges = states2edges(true_states)
    # add offset to account for local alignments
    true_edges = list(map(tuple, np.array(true_edges)))
    pred_edges = np.array(pred_edges)
    pred_edges[:, 0] += query_offset
    pred_edges[:, 1] += hit_offset
    pred_edges = list(map(tuple, pred_edges))

    res = []
    for k in kernel_widths:
        r = roc_edges_kernel_identity(true_edges, pred_edges, k)
        res.append(r)
    return res


def alignment_score(true_states: str, pred_states: str):
    """
    Computes ROC statistics on alignment
    Parameters
    ----------
    true_states : str
        Ground truth state string
    pred_states : str
        Predicted state string
    """
    pred_states = list(map(tmstate_f, pred_states))
    true_states = list(map(tmstate_f, true_states))
    pred_edges = states2edges(pred_states)
    true_edges = states2edges(true_states)
    stats = roc_edges(true_edges, pred_edges)
    return stats


def alignment_visualization(truth, pred, match, gap, xlen, ylen):
    """ Visualize alignment matrix

    Parameters
    ----------
    truth : torch.Tensor
        Ground truth alignment
    pred : torch.Tensor
        Predicted alignment
    match : torch.Tensor
        Match matrix
    gap : torch.Tensor
        Gap matrix
    xlen : int
        Length of protein x
    ylen : int
        Length of protein y

    Returns
    -------
    fig: matplotlib.pyplot.Figure
       Matplotlib figure
    ax : list of matplotlib.pyplot.Axes
       Matplotlib axes objects
    """
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax[0].imshow(truth[:xlen, :ylen], aspect='auto')
    ax[0].set_xlabel('Positions')
    ax[0].set_ylabel('Positions')
    ax[0].set_title('Ground truth alignment')
    im1 = ax[1].imshow(pred[:xlen, :ylen], aspect='auto')
    ax[1].set_xlabel('Positions')
    ax[1].set_title('Predicted alignment')
    fig.colorbar(im1, ax=ax[1])
    im2 = ax[2].imshow(match[:xlen, :ylen], aspect='auto')
    ax[2].set_xlabel('Positions')
    ax[2].set_title('Match scoring matrix')
    fig.colorbar(im2, ax=ax[2])
    im3 = ax[3].imshow(gap[:xlen, :ylen], aspect='auto')
    ax[3].set_xlabel('Positions')
    ax[3].set_title('Gap scoring matrix')
    fig.colorbar(im3, ax=ax[3])
    plt.tight_layout()
    return fig, ax


def alignment_text(x, y, pred, truth, stats):
    """ Used to visualize alignment as text

    Parameters
    ----------
    x : str
        Protein X
    y : str
        Protein Y
    pred : list of int
        Predicted states
    truth : list of int
        Ground truth states
    stats : list of float
        List of statistics from roc_edges
    """
    # TODO: we got the truth and prediction edges swapped somewhere earlier
    true_alignment = states2alignment(truth, x, y)
    pred_alignment = states2alignment(pred, x, y)
    cols = ['tp', 'fp', 'fn', 'perc_id', 'ppv', 'fnr', 'fdr']
    stats = list(map(lambda x: np.round(x, 2), stats))
    s = list(map(lambda x: f'{x[0]}: {x[1]}', list(zip(cols, stats))))

    stats_viz = ' '.join(s)
    truth_viz = (
        '# Ground truth\n'
        f'    {true_alignment[0]}\n    {true_alignment[1]}'
    )
    pred_viz = (
        '# Prediction\n'
        f'    {pred_alignment[0]}\n    {pred_alignment[1]}'
    )

    s = stats_viz + '\n' + truth_viz + '\n' + pred_viz
    return s
