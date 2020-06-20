import sys
import subprocess

fname = sys.argv[1]         # input PDB ID pairs
output = sys.argv[2]        # tm-align aggregated output
path = sys.argv[3]          # path to pdb
# by default it should be at
# /mnt/home/protfold/ceph/PDB_mirror/PDB/data/structures/divided/pdb
num_jobs = int(sys.argv[4])  # number of jobs to run concurrently

procs = []
for line in open(fname):
    x, y = line.rstrip().split(' ')

    x = x.lower()
    y = y.lower()
    xdiv = x[1:-1]
    ydiv = y[1:-1]
    xpath = f'{path}/{xdiv}/pdb{x}.ent.gz'
    ypath = f'{path}/{ydiv}/pdb{y}.ent.gz'
    xscratch = f'/scratch/pdb{x}.ent.gz'
    yscratch = f'/scratch/pdb{y}.ent.gz'
    xpdb = f'/scratch/pdb{x}.ent'
    ypdb = f'/scratch/pdb{y}.ent'
    cmd = '; '.join([
        f'cp {xpath} {xscratch}',
        f'cp {ypath} {yscratch}',
        f'gunzip {xscratch}',
        f'gunzip {yscratch}',
        f'TMalign {xpdb} {ypdb} >> {output}',
        f'rm {xpdb}; rm {ypdb}'
    ])

    proc = subprocess.Popen(cmd, shell=True)
    procs.append(proc)
    if len(procs) >= num_jobs:
        for p in procs:
            p.wait()
