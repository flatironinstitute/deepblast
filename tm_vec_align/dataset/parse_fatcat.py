import sys
import re


# This is used to parse the fatcat rigid output
def extract_f(x):
    if x[0] == 'd':
        y = x[1:]
    else:
        y = x.split(':')[1]
    return y[:4], y[4], y[5:]


for line in sys.stdin:
    tabs = re.split(r'\s+', line)
    id1, id2 = tabs[0], tabs[1]
    # parse out the pdb id and chains separately
    pdb1, chain1, _ = extract_f(id1)
    pdb2, chain2, _ = extract_f(id2)
    sys.stdout.write(f'{pdb1}\t{chain1}\t{pdb2}\t{chain2}\n')
