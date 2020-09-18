import os
from deepblast.dataset.utils import state_f, revstate_f
import pandas as pd
import numpy as np
from collections import Counter


def read_mali(root, tool='manual', report_ids=False):
    """ Reads in all alignments.

    Parameters
    ----------
    root : path
        Path to root directory
    tool : str
        Specifies which tools alignments should be extracted for.

    Returns
    -------
    pd.DataFrame
        Three columns, one for each sequence and the resulting alignment.
        If `report_ids` is specified, then the pdb id and the query/hit
        ids are also reported as additional columns.
    """
    res = []
    pdbs = []
    dirs = []
    for path, directories, files in os.walk(root):
        for f in files:
            if '.ali' in f and tool in f:
                fname = os.path.join(path, f)
                lines = open(fname).readlines()
                X = lines[0].rstrip().upper()
                Y = lines[1].rstrip().upper()
                S = ''.join(
                    list(map(revstate_f, map(state_f, list(zip(X, Y))))))
                res.append((X.replace('-', ''), Y.replace('-', ''), S))
                pdbs.append(os.path.basename(f).split('.')[0])
                dirs.append(os.path.basename(path))
    res = pd.DataFrame(res)
    if report_ids:
        res['query_id'] = np.arange(len(res)).astype(np.str)
        res['hit_id'] = (np.arange(len(res)) + len(res)).astype(np.str)
        res['pdb'] = pdbs
        res['dir'] = dirs

    return res


def _mammoth_strip(x):
    y = ''.join(x.split(' ')[1:])
    return y.rstrip()


def read_mali_mammoth(root, report_ids=False):
    """ Reads in all alignments.

    Parameters
    ----------
    root : path
        Path to root directory
    tool : str
        Specifies which tools alignments should be extracted for.

    Returns
    -------
    pd.DataFrame
        Three columns, one for each sequence and the resulting alignment.
        If `report_ids` is specified, then the pdb id and the query/hit
        ids are also reported as additional columns.
    """
    res = []
    pdbs = []
    for path, directories, files in os.walk(root):
        for f in files:

            if '.ali' in f:
                fname = os.path.join(path, f)
                contents = open(fname).readlines()
                pred = list(filter(lambda x: 'Prediction ' in x, contents))
                expr = list(filter(lambda x: 'Experiment ' in x, contents))
                idx = np.arange(len(pred)) % 2 == 0
                pred = list(np.array(pred)[idx])
                X = ''.join(list(map(_mammoth_strip, pred)))
                expr = list(np.array(expr)[~idx])
                Y = ''.join(list(map(_mammoth_strip, expr)))
                X, Y = X.replace('.', '-'), Y.replace('.', '-')
                X, Y = X.rstrip().upper(), Y.rstrip().upper()
                S = ''.join(
                    list(map(revstate_f, map(state_f, list(zip(X, Y))))))
                res.append((X.replace('-', ''), Y.replace('-', ''), S))
                pdbs.append(os.path.basename(f).split('.')[0])
    res = pd.DataFrame(res)
    if report_ids:
        res['query_id'] = np.arange(len(res)).astype(np.str)
        res['hit_id'] = (np.arange(len(res)) + len(res)).astype(np.str)
        res['pdb'] = pdbs
    return res


def get_mali_structure_stats(root):
    """ Reads in the manual alignments and obtains stats.

    Parameters
    ----------
    root : path
        Path to root directory

    Returns
    -------
    pd.DataFrame
        alpha residues
        beta residues
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP

    res = []
    tool = 'manual'
    for path, directories, files in os.walk(root):
        for f in files:
            if '.pdb' in f and tool in f:

                fname = os.path.join(path, f)
                parser = PDBParser()
                # ids = os.path.basename(fname).split('_')
                structs = parser.get_structure('', fname)
                dssp1 = DSSP(structs[0], fname, dssp='mkdssp')
                classes1 = list(map(lambda x: x[2], dssp1))
                len1 = len(classes1)

                classes1 = pd.Series(Counter(classes1))
                classes1.index = list(map(lambda x: 'x' + x, classes1.index))
                pdb_name = os.path.basename(f).split('.')[0]
                # stats = pd.concat((classes1, classes2))
                stats = classes1
                stats['pdb'] = pdb_name
                stats['path'] = fname
                stats['xlen'] = len1

                # dssp2 = DSSP(structs[1], fname, dssp='mkdssp')
                # classes2 = list(map(lambda x: x[2], dssp2))
                # len2 = len(classes2)
                # classes2 = pd.Series(Counter(classes2))
                # classes2.index = list(map(lambda x: 'y' + x, classes2.index))
                # stats['ylen'] = len2
                res.append(stats)

    res = pd.DataFrame(res)
    return res
