import numpy as np
import matplotlib.pyplot as plt
from deepblast.dataset.dataset import states2alignment


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
    ax[1].imshow(pred[:xlen, :ylen], aspect='auto')
    ax[1].set_xlabel('Positions')
    ax[1].set_title('Predicted alignment')
    ax[2].imshow(match[:xlen, :ylen], aspect='auto')
    ax[2].set_xlabel('Positions')
    ax[2].set_title('Match scoring matrix')
    ax[3].imshow(gap[:xlen, :ylen], aspect='auto')
    ax[3].set_xlabel('Positions')
    ax[3].set_title('Gap scoring matrix')
    return fig, ax


def alignment_text(x, y, pred, truth):
    """ Used to visualize alignment as text

    Parameters
    ----------
    x : str
        Protein X
    y : str
        Protein Y
    pred : list of it
        Predicted states
    truth : list of it
        Ground truth states
    """
    # TODO: we got the truth and prediction edges swapped somewhere earlier
    true_alignment = states2alignment(truth, x, y)
    pred_alignment = states2alignment(pred, x, y)

    truth_viz = (
        '# Ground truth\n'
        f'    {true_alignment[0]}\n    {true_alignment[1]}'
    )
    pred_viz = (
        '# Prediction\n'
        f'    {pred_alignment[0]}\n    {pred_alignment[1]}'
    )
    s = truth_viz + '\n' + pred_viz
    return s
