from deepblast.dataset.dataset import states2alignment
import matplotlib.pyplot as plt


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

def alignment_visualization(pred_aligment, truth_alignment):
    """ Visualize alignment matrix

    Parameters
    ----------
    pred_alignment : torch.Tensor
        Predicted alignment matrix
    truth_alignment : torch.Tensor
        Ground truth alignment matrix

    Returns
    -------
    fig: matplotlib.pyplot.Figure
       Matplotlib figure
    ax : list of matplotlib.pyplot.Axes
       Matplotlib axes objects
    """
    pred = pred_alignment.detach().cpu().numpy().squeeze()
    truth = truth_alignment.detach().cpu().numpy().squeeze()

    fig, ax = plt.subplots(3, 1)
    ax[0].imshow(pred)
    ax[0].set_xlabel('Positions')
    ax[0].set_ylabel('Positions')
    ax[0].set_title('Predicted \n alignment matrix')

    ax[2].imshow(pred)
    ax[2].set_xlabel('Positions')
    ax[2].set_ylabel('Positions')
    ax[2].set_title('Grouth truth\n alignment matrix')

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
    true_alignment = states2alignment(truth, y, x)
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
