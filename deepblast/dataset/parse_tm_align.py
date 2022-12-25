import re
import sys


""" Example of TM-align output versio 20170708
0 |
1 |  **************************************************************************
2 |  *                        TM-align (Version 20170708)                     *
3 |  * An algorithm for protein structure alignment and comparison            *
4 |  * Based on statistics:                                                   *
5 |  *       0.0 < TM-score < 0.30, random structural similarity              *
6 |  *       0.5 < TM-score < 1.00, in about the same fold                    *
7 |  * Reference: Y Zhang and J Skolnick, Nucl Acids Res 33, 2302-9 (2005)    *
8 |  * Please email your comments and suggestions to: zhng@umich.edu          *
9 |  **************************************************************************
10|
11| Name of Chain_1: /scratch/pdb3itf.ent
12| Name of Chain_2: /scratch/pdb2mce.ent
13| Length of Chain_1:  103 residues
14| Length of Chain_2:   21 residues
15|
16| Aligned length=   21, RMSD=   1.23, Seq_ID=n_identical/n_aligned= 0.048
17| TM-score= 0.18482 (if normalized by length of Chain_1)
18| TM-score= 0.28926 (if normalized by length of Chain_2)
19| (You should use TM-score normalized by length of the reference protein)
20|
21| (":" denotes aligned residue pairs of d < 5.0 A, "." denotes other aligned
22| STQSHFDGISLTEHQRQQRDLQQARHEQPPVNVSELETHRLVTAENFDENAVRAQAEKANEQIARQVEAKVRNQY
23|                                                :::::::::::::::::::::
24| -----------------------------------------------DAGHGQISHKRHKTDSFVGLM-------
"""


def aln_f(X):
    x, a, y = X
    if y == '-':
        return '1'
    if x == '-':
        return '2'
    else:
        return a


def parse_block_2017(lines):
    """
    Parameters
    ----------
    lines : list of str
       25 lines for tm-align output

    Returns
    -------
    chain1_name : str
       Name of chain 1
    chain2_name : str
       Name of chain 2
    tmscore1 : float
       TM score chain 1
    tmscore2 : float
       TM score for chain 2
    rmsd : float
       Root Mean Standard Deviation
    chain1 : str
       Protein string for chain1
    chain2 : str
       Protein string for chain2
    alignment : str
       Alignment string between chain 1 and chain2. This will output 4 states
       '1' : insertion in chain 1
       '2' : insertion in chain 2
       '.' : other aligned residues
       ':' : aligned residued within 5.0 A
    """
    chain1_name = lines[11].split(':')[1].rstrip().lstrip()
    chain2_name = lines[12].split(':')[1].rstrip().lstrip()
    tmscore1 = float(lines[17].lstrip().split(' ')[1])
    tmscore2 = float(lines[18].lstrip().split(' ')[1])
    chain1 = lines[22].rstrip().lstrip()
    aln = lines[23]
    chain2 = lines[24].rstrip().lstrip()
    rmsd = float(re.split(r'\s+', lines[16].lstrip().split(', ')[1])[1])
    zlist = list(zip(chain1, aln, chain2))
    alignment = ''.join(list(map(aln_f, zlist)))
    chain1 = chain1.replace('-', '')
    chain2 = chain2.replace('-', '')
    return (chain1_name, chain2_name, tmscore1, tmscore2, rmsd,
            chain1, chain2, alignment)



""" Example of TM-align output version 20210224
 0|
 1|  *********************************************************************
 2|  * TM-align (Version 20210224): protein structure alignment          *
 3|  * References: Y Zhang, J Skolnick. Nucl Acids Res 33, 2302-9 (2005) *
 4|  * Please email comments and suggestions to yangzhanglab@umich.edu   *
 5|  *********************************************************************
 6|
 7| Name of Chain_1: /mnt/ceph/users/protfold/deepfri_and_friends/swissmodel/annot/raw/structures/Q0/Q0VRZ6.pdb (to be superimposed onto Chain_2)
 8| Name of Chain_2: /mnt/ceph/users/protfold/deepfri_and_friends/swissmodel/annot/raw/structures/Q3/Q3MH15.pdb
 9| Length of Chain_1: 299 residues
10| Length of Chain_2: 27 residues
11|
12| Aligned length= 20, RMSD=   1.89, Seq_ID=n_identical/n_aligned= 0.050
13| TM-score= 0.06204 (if normalized by length of Chain_1, i.e., LN=299, d0=6.35)
14| TM-score= 0.43755 (if normalized by length of Chain_2, i.e., LN=27, d0=1.04)
15| (You should use TM-score normalized by length of the reference structure)
16|
17| (":" denotes residue pairs of d <  5.0 Angstrom, "." denotes other aligned residues)
18| MIRQRTLKNVIRATGVGLHTGEKVYMTVRPAPVDTGIVFRRVDLDPVVEIKAAADAVGETTLSSTLVQDGVKVGTVEHFLSAMAGLGIDNAFVELSAPEMPIMDGSAGPFVFLLQSAGIKEQEAAKKFIRIKKEVTVREDDKTATFVPFDGFKVTFSIEFDHPVFEERNQLASIDFSTTSFVKEVARARTFGFMRDIEFLRSQNLALGGSVDNAIVVDEYRILNEDGLRYDDEFVKHKMLDAI>
19|                                                                                                                                                                                                                                           :::::::::>
20| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------NEPVRHKIL>
21|
"""

def validate_block_2021(lines):
    chain1 = 'Chain_1' in lines[7]
    chain2 = 'Chain_2' in lines[8]
    tm1 = 'TM-score' in lines[13]
    tm2 = 'TM-score' in lines[14]
    aln = ':' in lines[19]
    # alphabet = 'ARNDCQEGHILKMFPSTWYVXOUBZ-'
    # seq1 = any([a in alphabet for a in lines[18]])
    # seq2 = any([a in alphabet for a in lines[19]])
    seq1 = ' ' not in lines[18].strip()
    seq2 = ' ' not in lines[20].strip()
    rmsd = 'RMSD' in lines[12]
    return chain1 and chain2 and tm1 and tm2 and aln and rmsd and seq1 and seq2

def parse_block_2021(lines):
    """
    Parameters
    ----------
    lines : list of str
       25 lines for tm-align output

    Returns
    -------
    chain1_name : str
       Name of chain 1
    chain2_name : str
       Name of chain 2
    tmscore1 : float
       TM score chain 1
    tmscore2 : float
       TM score for chain 2
    rmsd : float
       Root Mean Standard Deviation
    chain1 : str
       Protein string for chain1
    chain2 : str
       Protein string for chain2
    alignment : str
       Alignment string between chain 1 and chain2. This will output 4 states
       '1' : insertion in chain 1
       '2' : insertion in chain 2
       '.' : other aligned residues
       ':' : aligned residued within 5.0 A
    """
    chain1_name = lines[7].split(':')[1].rstrip().lstrip().split(' ')[0]
    chain2_name = lines[8].split(':')[1].rstrip().lstrip().split(' ')[0]
    print(chain1_name, chain2_name)
    tmscore1 = float(lines[13].lstrip().split(' ')[1])
    tmscore2 = float(lines[14].lstrip().split(' ')[1])
    chain1 = lines[18].rstrip().lstrip()
    aln = lines[19]
    chain2 = lines[20].rstrip().lstrip()
    rmsd = float(re.split(r'\s+', lines[12].lstrip().split(', ')[1])[1])
    zlist = list(zip(chain1, aln, chain2))
    alignment = ''.join(list(map(aln_f, zlist)))
    chain1 = chain1.replace('-', '')
    chain2 = chain2.replace('-', '')
    return (chain1_name, chain2_name, tmscore1, tmscore2, rmsd,
            chain1, chain2, alignment)


if __name__ == '__main__':
    lines_per_block = 23
    fname = sys.argv[1]      # tm-align output name
    output = sys.argv[2]     # output tab delimited file
    block = []
    i = 0
    with open(output, 'w') as outhandle:
        for line in open(fname):
            #if 'Total CPU time' in line:
            #    continue

            if i % lines_per_block == 0 and i > 0:
                val = validate_block_2021(block)
                # print(''.join(block))
                if val:
                    res = parse_block_2021(block)
                    block = []
                    (chain1_name, chain2_name, tmscore1, tmscore2, rmsd,
                     chain1, chain2, alignment) = res
                    s = '\t'.join([f'{chain1_name}', f'{chain2_name}',
                                   f'{tmscore1}', f'{tmscore2}',
                                   f'{rmsd}',
                                   f'{chain1}', f'{chain2}',
                                   f'{alignment}'])
                    outhandle.write(f'{s}\n')
                else:
                    i = i - 1
                    # pop front element
                    block = block[1:]

            block.append(line)
            i += 1
