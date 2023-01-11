import numpy as np
from collections import namedtuple
from deepblast.dataset.parse_pdb import readPDB
import pickle
import warnings


def kabsch_modify(points1, points2, centerCOM=True, epsilon=1E-16):
    ''' Modified version of Kabsch algorithm

    Parameters
    ----------
    point1 : np.array
        Point cloud of N d-dimensional points
    points2 : np.array
        Point cloud of N d-dimensional points
    centerCOM : bool
        Specifies if centering will be performed on the two point clouds
    episilon : float
        Rounding error to check for centering

    Returns
    -------
    R : np.array
       Rotation matrix
    w : np.array
       Lengths along principal directions
    d : np.array
       Sign directionality

    Notes
    -----
    The kabsch method requires centered arrays and thus perfroms this centering
    in-place unless the COM position <epsilon (i.e. close enough to zero).
    But if you already centered them beforehand the optionally overide this
    by setting centerCOM=False.

    To get the RMSD then apply the rotation like this:
    np.sum((np.matmul(p1,R)-p2)**2) to rotate p1 onto p2
    np.sum((np.matmul(p2,R.T)-p1)**2)to rotate p2 onto p1
    First Author: Charlie E. M Strauss 2020'''

    if centerCOM:
        t = np.mean(points1,axis=0)
        if np.any(abs(t)>epsilon):
            points1-= t
        t = np.mean(points2,axis=0)
        if np.any(abs(t)>epsilon):
            points2-= t

    H =  np.matmul(points1.T,points2)
    V, w, U = np.linalg.svd(H)

    R = np.matmul(V,U)
    # check to make sure rotation is proper and not an inversion.
    # this happens mainly when structures are not similar, but it does happen!
    d = np.sign(np.linalg.det(R))

    if d == -1:
        # fix improper rotation.  It's not intuitvley obvious why the best
        # proper rotation is a sign flip on U but it is. how convenient!
         U[-1, :]=-U[-1, :]   # note!  or NOT this:  U[:,-1]
         R = np.matmul(V, U)

    return R, w, d


def kabsch(p1, p2):
    ''' Kabsch translation and rotation algorithm.

    Parameters
    ----------
    p1 : np.array
        Point cloud of N d-dimensional points
    p2 : np.array
        Point cloud of N d-dimensional points

    Returns
    -------
    R : np.array
       Rotation matrix
    w : np.array
       Lengths along principal directions
    d : np.array
       Sign directionality
    offset1 : np.array
       Translation bias for points1
    offset2 : np.array
       Translation bias for points2

    Notes
    -----
    To get the RMSD then apply the rotation like this:
        np.sum((np.matmul(p1,R)-p2)**2) to rotate p1 onto p2
        np.sum((np.matmul(p2,R.T)-p1)**2)to rotate p2 onto p1
    First Author: Charlie E. M Strauss 2020'''

    # center both arrays on COM
    # this also makes a copy so as not to modify inputs
    offset1 = np.mean(p1, axis=0)
    offset2 = np.mean(p2, axis=0)
    points1 = p1 - offset1
    points2 = p2 - offset2
    R, w, d = kabsch_modify(points1,points2,centerCOM=False)
    return (R, w, d, offset1, offset2)


def kabsch_template_alignment(p1, p2, t1, t2):
    ''' Given a matched pair of templates, it computes the optimal
        superposition rotation and translation this is then applied
        to the coordinates in P1 and P2. the inputs are not modified

    Parameters
    ----------
    p1 : np.array
       Point cloud of N d-dimensional points
    p2 : np.array
       Point cloud of M d-dimensional points

    Returns
    -------
    p1_new : np.array
       Translated point cloud p1
    p2_new : np.array
       Translated and rotated point cloud p2
    params : tuple of array-like
       Rotation and translation from kabsch superposition algorithm
    '''
    R, w, d, offset1, offset2 = kabsch(t1, t2)
    p1_new = p1 - offset1
    p2_new = np.matmul(p2 - offset2,R.T)
    params = (R, w, d, offset1, offset2)
    return p1_new, p2_new, params


MAXSUB_TM= namedtuple('MAXSUB_TM',('score','rotation','alignment','alignedRMS'))


def FR_TM_maxsub_score(master_p0, master_p1,align_index,
          FRAGSMALL=8,FRAGLARGE=12,TOL=7.0, UNIT = 1.0 ):
    ''' Computes TM-scores

    Parameters
    ----------
    p1 : np.array
       Point cloud of N d-dimensional points
    p2 : np.array
       Point cloud of M d-dimensional points
    align_index : np.array
       Translates i'th alingment point to the index in master

    Returns
    -------
    Tuple of metrics
    '''
    RMSTOL = TOL * UNIT   #######  CHECK THIS IS THE RIGHT DEFAULT VALUE
    L_aligned=np.shape(align_index)[1]
    L_min = min( np.shape(master_p0)[0],np.shape(master_p0)[0])  # get the shape
    assert(L_min >9)

    TM_d0 = 1.24*(L_min-15)**0.333333 -1.8  # for the TM score
    TM_d02 = TM_d0**2
    FRAGSIZE =FRAGSMALL if L_min <100 else FRAGLARGE


    N = np.shape(align_index)[1]
    WINDOWS = N - FRAGSIZE  # +1 ?
    # grab the aligned residues out of the master lists.
    p0 = master_p0[align_index[0]]
    p1 = master_p1[align_index[1]]

    # set accumulation variables to default or triggers to replace them later
    maxsub_most = -1 # or  0?
    maxsub_alignedRMS = 1E9*UNIT
    maxsub_alignment=[]
    maxsub_rotation = np.eye(3)

    raw_TM_score_best = maxsub_TM_score_best = 1E9 * UNIT

    # initialize
    # an illegally large value so it will get forced to be replaced below
    raw_TM_score_best = -1
    maxsub_TM_best_rotation=raw_TM_best_rotation = np.array(
        [[1.0,0,0],[0,1.0,0.0],[0,0,1.0]]) # no rotation at all.
    maxsub_TM_score_best = -1

    # loop over all initial seeds for rotation based on consecutive fragments.

    for i0 in range( WINDOWS):
        frag0 = p0[i0 : i0 + FRAGSIZE]
        frag1 = p1[i0 : i0 + FRAGSIZE]
        p0aligned,p1aligned,G=kabsch_template_alignment(p0,p1,frag0,frag1)
        # The initial alginment  might be okay, so lets record it
        deviation2 = np.sum((p0aligned-p1aligned)**2,axis=1) # only sum over x,y,z but not over points.
        raw_TM_score_temp = np.sum(1.0/(1.0+deviation2/TM_d02 ))/L_min # can use L_target instead of L_min for one-to-many
        raw_rmsd_temp= np.sqrt(np.mean(deviation2))
        if  raw_rmsd_temp>=RMSTOL : next  # don't even bother!  (not a problem for mammoth but is for arbitrary alignments)

        if  raw_TM_score_temp> raw_TM_score_best:
             raw_TM_score_best = raw_TM_score_temp
             raw_TM_best_rotation = G
             raw_TM_best_seed_alignment = range(i0,i0+FRAGSIZE)
             raw_TM_alignedRMS = raw_rmsd_temp

        if  raw_TM_score_temp > maxsub_TM_score_best:
             maxsub_TM_score_best = raw_TM_score_temp
             maxsub_TM_best_rotation = G
             maxsub_TM_best_seed_alignment = range(i0,i0+FRAGSIZE)
             maxsub_TM_alignedRMS = raw_rmsd_temp
        # now lets apply some maxsub style iterations to improve this
        last_pair_count = 0
        indicies= []  # list(fragment_indices) # copy current fragment indicies in list
        #### note maybe we should insert the fragment because if t==0.1 then some of the fragment won'g go in!!!!!!
        #########################
        TM_temp = 0.0
        # for TM_align it makes sense to make TOL larger since we want it to keep searching longer.
        t = 0.0  #################  nayve we should start this at 1???/
        while t < TOL:  # this goes one unit over TOL.  should we fix that?
            t += 0.1    #0.25*UNIT
            t2 = t * t  # squared radius
            # this embeds a logic error in mammoth!  need to fix that!
            # The percentage of structural similarity (PSI) is defined as the number of
            # amino acid pairs with Cα atoms that are closer in space then 4 Å after
            # problem happens if NO residues get included on first pass
            # then the t gets raised too much!
            min_d2 = (TOL+UNIT)*(TOL+UNIT)
            for j0 in range(N):
                # check if we should include this atom pair
                if j0 not in indicies:
                    deviations = (p0aligned[j0] - p1aligned[j0]) ** 2
                    d2 = np.sum(deviations)  # MSD for atom pair so it's divide by 1 not L_align
                    # include this if it is close enough or is in the fragment
                    if d2 < t2 or (0< =j 0 - i0 < FRAGSIZE): # could use short circuit or
                        indicies.append(j0)  # change this to faster method
                    else:  # do not update min_d2 this for fragment itself
                        min_d2 = min(min_d2, d2) # track closest distance
                        # note this min ignores the ones in the aligned fragment.
            # did we add any new atomns in last iteration?
            L_indicies = len(indicies)

            # here we have to be careful--- as long as indicies is strictly accumaltive then we can be sure
            # that nothing in the alignemnt was added unless  L_indicies > last_pair_count
            # but if that is NOT true, and we can remove underperforming residues as the alignment focus
            # moves to better TM scores but fewer residues then we may not trigger this
            # since inside this we trigger the re-alignment then the evolution will stop in it's tracks!
            if  L_indicies > last_pair_count and  L_indicies > 3:
                last_pair_count= L_indicies
                p0aligned,p1aligned,G=kabsch_template_alignment(p0,p1,p0[indicies],p1[indicies])
                # rotation that aligns just the selected atoms
                deviation2 = np.sum((p0aligned-p1aligned)**2,axis=1)
                alignedRMS = np.sqrt(np.mean(deviation2))  # (assumes no replicates!)

                # MAXSUB SCORED
                # for MAXSUB, we prefer greater number of aligned residues as long as rms<RMSTOL
                # and amoung equal number of residues we take the smaller rms.
                if (L_indicies > maxsub_most and alignedRMS <= RMSTOL) or \
                    (L_indicies == maxsub_most and alignedRMS < maxsub_alignedRMS):
                    # update the best we have0 72 seen so far
                    maxsub_most = L_indicies
                    maxsub_alignedRMS= alignedRMS
                    maxsub_alignment= np.array(indicies)
                    maxsub_rotation = G

                #TM SCORED
                maxsub_TM_score_temp = np.sum(1.0/(1.0+deviation2/TM_d02 ))/L_min  # can use L_target instead of L_min for one-to-many
                # it feels wrong to divide by L_min and not L_align because it automatically upper bounds the TM_score to L_align/L_min
                if  maxsub_TM_score_temp > maxsub_TM_score_best:
                     maxsub_TM_score_best = maxsub_TM_score_temp
                     maxsub_TM_best_rotation = G
                     maxsub_TM_best_seed_alignment = np.array(indicies)
                     maxsub_TM_alignedRMS = alignedRMS
                     # be careful with the logic of which array to rotate
            else:  # nothing was close enough at this tolerance
                t = np.sqrt(min_d2)# Will also add to this 0.1*UNIT above

        L_indicies = len(indicies)

        alignedRMS = np.sqrt(np.sum((p0aligned[indicies] - p1aligned[indicies]) ** 2) / L_indicies)


        best = MAXSUB_TM(maxsub_TM_score_best,maxsub_TM_best_rotation,maxsub_TM_best_seed_alignment,maxsub_TM_alignedRMS)
        raw = MAXSUB_TM (raw_TM_score_best,raw_TM_best_rotation,raw_TM_best_seed_alignment,raw_TM_alignedRMS)
        most = MAXSUB_TM(maxsub_most,maxsub_rotation,maxsub_alignment,maxsub_alignedRMS)

    return best, raw, most


Metrics = namedtuple('Metrics', ['TM','PSI','aPSI','oPSI','rPSI',
                                 'cRMS','aRMS','oRMS',
                                 'aSeq_ident','oSeq_ident','cSeq_Ident',
                                 'L_min','L_aligned','L_orientable','L_PSI'])



def standard_metrics(master_p0, master_p1, align_index, indicies, d0=4.0, UNIT=1.0):
    ''' Compute multiple structural alignment goodness-of-fit metrics

    Parameters
    ----------
    p1 : np.array
       Point cloud of N d-dimensional points
    p2 : np.array
       Point cloud of M d-dimensional points
    align_index : np.array
       Translates i'th alingment point to the index in master

    Returns
    -------
    Tuple of metrics

    Notes
    -----
    The percentage of structural similarity (PSI) is defined as the number of aligned
    amino acid pairs with Cα atoms that are closer in space then 4 Å after optimal superposition
    normalized by the length of the shorter chain in the alignment.
    The relevant PSI (rPSI) value does not include fragments shorter than four aligned amino acid
    from the calculated PSI value. The coordinate RMSD (cRMSD) is computed for all aligned pairs
    after optimal superposition. The cRMSD (core) is computed for those aligned pairs that
    contribute to the PSI value [35]. Here, PSI/rPSI provides a more detailed view of the
    alignment. However, these values are length dependent.'''
    if indicies is None:
        indicies = range(np.shape(align_index)[1])
    L_min = min( np.shape(master_p0)[0],np.shape(master_p1)[0])  # get the shape
    L_aligned=np.shape(align_index)[1]
    L_orientable = len(indicies)

    p0 = master_p0[align_index[0]]
    p1 = master_p1[align_index[1]]


    p0aligned, p1aligned,G= kabsch_template_alignment(p0, p1, p0[indicies], p1[indicies])
    # rotation aligns just the selected atoms
    deviation2 = np.sum((p0aligned - p1aligned) ** 2,axis=1)  # this is over the full alignment.
    RMS = np.sqrt(np.sum(deviation2) / L_aligned)             #RMS over all
    oRMS = np.sqrt(np.sum(deviation2[indicies]) / L_orientable)  # RMS over subset of resides contributing to orientation

    PSI_mask =  deviation2 < (4.0 * UNIT) ** 2
    L_PSI = np.sum(PSI_mask)
    PSI= L_PSI / L_min

    if L_PSI > 2: # RMS doesn't mean much otherwise
        # RMS over resides contributing to PSI
        cRMS = np.sqrt(np.sum(deviation2[PSI_mask])/L_PSI)  # or should be re-align this?  could become tailchasing?
    else:
        cRMS = 100.0  # fiction.

    c=0  # run length tracker
    rPSI=0  # running sum of long runs with 4 or more residues in a row without gaps
    for i in range( L_aligned-1):
        c += 1
        if np.any((align_index[:, i+1] - align_index[:, i]) > 1):  # a gap in either protein's alignment
           if c > 3: rPSI += c  # should I be checking for 3 or 4 here?
           c= 0
    rPSI = rPSI / L_min
    return Metrics(PSI, rPSI, cRMS, RMS, oRMS, L_min, L_aligned, L_orientable, L_PSI)


def parseAlingmentString(j):
    """This parses the deepblast alignment string,
    but this is superceded by another method
    this one returns the weak alignment but not the strong alignment.
    however for now this the only way to access the deepBlast alignment.

    Parameters
    ----------
    j : str
       Alignment string

    Returns
    -------
    np.array
       Two set of alignment indices
    """
    c0 = 0
    c1 = 0
    a00 = []
    a01 = []
    for i in j:
        if i == ':':
            a00.append(c0)
            c0+=1
            a01.append(c1)
            c1+=1
        elif i == '1':
            c0+=1
        elif i == '2':
            c1+=1
    return np.array([a01, a00])


def process_alignment(alignment, seq0, seq1, pdb0, pdb1):
    """ Processes a single alignment

    Parameters
    ----------
    alignments : str
       Alignment string
    seq0 : str
       First sequence
    seq1 = str
       Second sequence
    pdb0 : path
       Path to first PDB file
    pdb1 : path
       Path to seconde PDB file

    Returns
    -------
    tuple : standard_metrics
    """
    fpnts0 = readPDB(pdb0)
    fpnts1 = readPDB(pdb1)
    a1 = parseAlingmentString(alignment)
    if (fpnts0.seq != seq0 or fpnts1.seq != seq1):
        if fpnts0.seq != seq0:
            warnings.warn(
                "sequence {} does not match pdb {}".format(seq0, pdb0))
        if fpnts1.seq != seq1:
            warnings.warn(
                "sequence {} does not match pdb {}".format(seq1, pdb1))

    A, B, C = FR_TM_maxsub_score(fpnts0.CA, fpnts1.CA, a1)
    sm = standard_metrics(fpnts0.CA, fpnts1.CA, a1,
                          indicies=A.alignment,
                          seq0=seq0,
                          seq1=seq1,
                          d0=4.0, UNIT=1.0)
    return sm
