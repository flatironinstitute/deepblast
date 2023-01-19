import numpy as np
from collections import namedtuple
import warnings


PDBATOM = namedtuple('PDBATOM', ['tag', 'serial', 'atomname', 'altchar',
                                 'resname', 'chain', 'seqnum', 'insert',
                                 'x', 'y', 'z', 'occupancy', 'temp',
                                 'element', 'charge'])
PDBCA = namedtuple('PDBCA', ['seq', 'CA', 'first_resnum', 'length'])
aaname3 = np.array(['ALA', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS',
                    'ILE', 'LYS', 'LEU', 'MET', 'ASN', 'PRO', 'GLN',
                    'ARG', 'SER', 'THR', 'VAL', 'TRP', 'TYR', 'XXX'])
aaname1 = np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
                    'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'])

def readPDB(filename,verbose=False):
    ''' Reads a simple pdb file.  it only reads the first chain if there are more than one.  It extracts the sequence and Calpha
    records from it.  It does a little light reality checking to make sure things are consistent, but doesnt go through every
    possible common malfunctions one finds in a pdb file as this is just a lightweight reader.
    The return types include a namedtuple data structure that is specified in the code'''
    points = []
    seq = ''
    aa321 = dict()
    for tt1,tt3 in zip(aaname1,aaname3):  aa321[tt3]=tt1
    last_res_num = None
    flag=False
    print("read PDB file {}".format(filename))
    with open(filename) as pdbfile:
        flag=True
        for line in pdbfile:
            if line[:3] == 'TER'  or line[:6] == 'ENDMDL': break
            if line[:4] == 'ATOM' or line[:6] == "HETATM":
                # Split the line
                rec = PDBATOM(line[:6], int(line[6:11]), line[12:16], line[16],line[17:20],line[21],
                              int(line[22:26]),line[26],
                              float(line[30:38]),float(line[38:46]), float(line[46:54]),
                              line[54:60],line[60:66],line[76:78],line[78:80])

                if rec.atomname==' CA ':
                    #print(line,end='')
                    if last_res_num == None:
                        first_res_num=rec.seqnum
                        last_res_num=rec.seqnum-1
                    if rec.seqnum-last_res_num != 1:
                        print ("ERROR: {} missing or duplicate residue {}".format(filename,rec.seqnum))
                        flag=False
                    last_res_num=rec.seqnum
                    # To format again the pdb file with the fields extracted

                    if verbose: print ("%-6s%5s %4s%1s%3s %1s%4s%1s   %8s%8s%8s%6s%6s          %2s%2s"%rec)
                    if rec.resname in aa321:
                        h=aa321[rec.resname]
                    else:
                        h = 'Z'
                        print ("WARNING: unknown residue type ", rec.resname," ",rec.seqnum)
                        print ("%3d %1s %3s %8.3f %8.3f %8.3f\n"%(int(first_res_num),h,rec.resname,rec.x,rec.y,rec.z))
                   # print(rec)
                        flag=False
                    if rec.resname=="XXX": print ("warning: Fake residue XXX type detected at",rec.seqnum)
                    points.append([rec.x,rec.y,rec.z])
                    seq+=h

    if not flag: print ("there was an error")
    print("CA count {}".format(len(points)))
    return flag,PDBCA(seq,np.array(points),first_res_num, len(seq))
