import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
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


def alignment_visualization(truth_alignment, pred_alignment, pred_matrix):
    """ Visualize alignment matrix

    Parameters
    ----------
    truth_alignment : torch.Tensor
        Ground truth alignment
    pred_alignment : list of tuple
        Predicted alignment
    pred_matrix : list of tuple
        Predicted alignment matrix

    Returns
    -------
    fig: matplotlib.pyplot.Figure
       Matplotlib figure
    ax : list of matplotlib.pyplot.Axes
       Matplotlib axes objects
    """
    px, py = list(zip(*pred_alignment))
    tx, ty = list(zip(*truth_alignment))
    pred_matrix = pred_matrix.detach().cpu().numpy().squeeze()
    pred = coo_matrix((np.ones(len(pred_alignment)),
                       (px, py))).todense()

    truth = coo_matrix((np.ones(len(truth_alignment)),
                        (tx, ty))).todense()
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    ax[0].imshow(truth)
    ax[0].set_xlabel('Positions')
    ax[0].set_ylabel('Positions')
    ax[0].set_title('Ground truth alignment')
    ax[1].imshow(pred)
    ax[1].set_xlabel('Positions')
    ax[1].set_ylabel('Positions')
    ax[1].set_title('Predicted alignment')
    ax[2].imshow(pred_matrix)
    ax[2].set_xlabel('Positions')
    ax[2].set_ylabel('Positions')
    ax[2].set_title('Predicted alignment matrix')
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
    true_alignment = states2alignment(truth, y, x)
    pred_alignment = states2alignment(pred, x, y)

    truth_viz = (
        '# Ground truth\n'
        f'    {true_alignment[1]}\n    {true_alignment[0]}'
    )
    pred_viz = (
        '# Prediction\n'
        f'    {pred_alignment[0]}\n    {pred_alignment[1]}'
    )
    s = truth_viz + '\n' + pred_viz
    return s
