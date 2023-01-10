from deepblast.dataset.utils import state_f, revstate_f
from deepblast.dataset.parse_mali import read_mali
from Bio import SearchIO
import numpy as np
import pandas as pd


def parse_hmmer_text(hmmer_path):
    records = SearchIO.parse(hmmer_path, 'hmmer3-text')
    results = []
    for idx, cur in enumerate(records):
        for hit in cur.hits:
            for i, hsp in enumerate(hit.hsps):
                if cur.id != hit.id:
                    qs = hsp.fragment.query_start
                    qe = hsp.fragment.query_end
                    he = hsp.fragment.hit_end
                    hs = hsp.fragment.hit_start
                    query_s = str(hsp.fragment.query.seq)
                    hit_s = str(hsp.fragment.hit.seq)
                    score = hsp.bitscore
                    expect = hsp.evalue
                    toks = list(
                        map(str, [cur.id, hit.id, i, qs, qe, hs, he,
                                  query_s, hit_s, score, expect]))
                    results.append(toks)
    columns = ['query_id', 'hit_id', 'fragment_num',
               'query_start', 'query_end', 'hit_start', 'hit_end',
               'query_string', 'hit_string', 'score', 'evalue']
    return pd.DataFrame(results, columns=columns)


def hit_argmax(x):
    i = np.argmin(x['evalue'])
    return x.iloc[i]


def get_hmmer_alignments(hmmer_path, mali_root):
    hmmer_df = parse_hmmer_text(hmmer_path)
    # only consider the top scoring local alignments
    hmmer_df = hmmer_df.groupby(['query_id', 'hit_id']).apply(hit_argmax)
    manual = read_mali(mali_root, tool='manual', report_ids=True)
    idx = set(map(tuple, manual[['query_id', 'hit_id']].values))
    bidx = set(list(hmmer_df.index))
    idx = list(set(idx) & set(bidx))
    hmmer_df = hmmer_df.loc[idx]

    states = list(map(state_f, zip(list(hmmer_df['query_string']),
                                   list(hmmer_df['hit_string']))))
    states = ''.join(list(map(revstate_f, states)))
    hmmer_df['aln'] = states
    return hmmer_df
