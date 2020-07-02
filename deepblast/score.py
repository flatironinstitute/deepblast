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


def alignment_visualization(x, y, pred, truth):
    """
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
    if len(x) > len(y):
        true_alignment = states2alignment(truth, y, x)
        pred_alignment = states2alignment(pred, y, x)
    else:
        true_alignment = states2alignment(truth, x, y)
        pred_alignment = states2alignment(pred, x, y)

    truth_viz = (
        '# Ground truth\n'
        f'{true_alignment[0]}\n{true_alignment[1]}'
    )
    pred_viz = (
        '# Prediction\n'
        f'{pred_alignment[0]}\n{pred_alignment[1]}'
    )
    return truth_viz + '\n' + pred_viz
