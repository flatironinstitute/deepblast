from subprocess import Popen, PIPE
import re
from random import randint
import pandas as pd


def genpairs(n):
    seen = set()
    x, y = randint(0, n - 1), randint(0, n - 1)
    while True:
        seen.add((x, y))
        yield (x, y)
        x, y = randint(0, n - 1), randint(0, n - 1)
        while (x, y) in seen and (x == y):
            x, y = randint(0, n - 1), randint(0, n - 1)


def match(x):
    i, j = x
    if i == '.' and j == '.':
        return ''
    else:
        return (i, j)


def state_f(x):
    i, j = x
    if i == '.' and j != '.':
        return '1'
    if i != '.' and j == '.':
        return '2'
    else:
        return ':'


def parse_alignment(ai, aj):
    alignment = list(zip(list(ai), list(aj)))
    x, y = zip(*alignment)
    states = ''.join(list(map(state_f, alignment)))
    return ''.join(x), ''.join(y), states


def gen_alignments(msa, n_alignments):
    gen = genpairs(len(msa))
    alignments = []
    for k in range(n_alignments):
        i, j = next(gen)
        n1, ai = re.split(r'\s+', msa[i])
        n2, aj = re.split(r'\s+', msa[j])
        x, y, s = parse_alignment(ai, aj)
        alignments.append((n1, n2, 0, 0, 0, x, y, s))
    return alignments

def hmm_alignments(n, seed, n_alignments, hmmfile):
    cmd = f'hmmemit -a -N {n} --seed {seed} {hmmfile}'
    proc = Popen(cmd, shell=True, stdout=PIPE)
    # proc.wait()
    # construct alignments
    lines = proc.stdout.readlines()
    lines = list(map(lambda x: x.decode('utf-8'), lines))
    lines = list(map(lambda x: x.rstrip().upper(), lines))
    alignments = gen_alignments(lines[3:-2], n_alignments)
    df = pd.DataFrame(alignments)
    return df
