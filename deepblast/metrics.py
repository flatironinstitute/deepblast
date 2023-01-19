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
          FRAGSMALL=8,FRAGLARGE=12,TOL=7.0, UNIT = 1.0, debug=0 ):
    '''Inputs:
    master_pX Nxk points list where k=3 for 3D, and N is all points.
    align_index[X][i] translates i'th alingment point to the index in master

    '''
    RMSTOL = TOL*UNIT   #######  CHECK THIS IS THE RIGHT DEFAULT VALUE
    L_aligned=np.shape(align_index)[1]
    L_min = min( np.shape(master_p0)[0],np.shape(master_p1)[0])  # get the shape
    assert(L_min >9)

    TM_d0 = 1.24*(L_min-15)**0.333333 -1.8  # for the TM score
    TM_d02 = TM_d0**2
    FRAGSIZE =FRAGSMALL if L_min <100 else FRAGLARGE
    FRAGSIZE=7

    N = np.shape(align_index)[1]
    WINDOWS = N-FRAGSIZE # +1 ?
    # grab the aligned residues out of the master lists.
 #   print("P0 shape {} {}".format(np.shape(master_p0),align_index[0,-1]))
 #   print("P1 shape {} {}".format(np.shape(master_p1),align_index[1,-1]))
 #   print(align_index)
    p0 = master_p0[align_index[0]]
    p1 = master_p1[align_index[1]]


    # set accumulation variables to default or triggers to replace them later
    maxsub_most = -1 # or  0?
    maxsub_alignedRMS = 1E9*UNIT
    maxsub_alignment=[]
    maxsub_rotation = np.eye(3)

    raw_TM_score_best=maxsub_TM_score_best=1E9*UNIT


    # initialize
    raw_TM_score_best = -1 # an illegally large value so it will get forced to be replaced below
    maxsub_TM_best_rotation=raw_TM_best_rotation = np.array([[1.0,0,0],[0,1.0,0.0],[0,0,1.0]]) # no rotation at all.
    maxsub_TM_score_best = -1
    maxsub_TM_most = -1

    longest_TM_best_rotation=raw_TM_best_rotation = np.array([[1.0,0,0],[0,1.0,0.0],[0,0,1.0]]) # no rotation at all.
    longest_TM_score_best = -1
    longest_TM_most = -1

    # loop over all initial seeds for rotation based on consecutive fragments.

    for i0 in range( WINDOWS):
        frg = range(i0,i0+FRAGSIZE)
        frag0 = p0[frg]
        frag1 = p1[frg]

        p0aligned,p1aligned,G=kabsch_template_alignment(p0,p1,frag0,frag1)
      #  R,w,d,offset0,offset1 = kabsch(frag0,frag1) # just align this fragment

       # p1aligned = np.matmul(p1-offset1,R.T) # apply alignment.

        # The initial alginment  might be okay, so lets record it
        deviation2 = np.sum((p0aligned-p1aligned)**2,axis=1) # only sum over x,y,z but not over points.
        raw_TM_score_temp = np.sum(1.0/(1.0+deviation2/TM_d02 ))/L_min # can use L_target instead of L_min for one-to-many
    #    print ("iitial fragment msd ",np.sum(deviation2)/FRAGSIZE)
    #    print (align_index[:,i0:i0+FRAGSIZE])
        raw_rmsd_temp= np.sqrt(np.mean(deviation2))
        if  raw_rmsd_temp>=RMSTOL : next  # don't even bother!  (not a problem for mammoth but is for arbitrary alignments)


        if  raw_TM_score_temp> raw_TM_score_best:
            raw_TM_score_best = raw_TM_score_temp
            raw_TM_best_rotation = G
            raw_TM_best_seed_alignment = frg
            raw_TM_alignedRMS=raw_rmsd_temp

        if  raw_TM_score_temp> maxsub_TM_score_best:
            maxsub_TM_score_best = raw_TM_score_temp
            maxsub_TM_best_rotation = G
            maxsub_TM_best_seed_alignment = frg
            maxsub_TM_alignedRMS = raw_rmsd_temp


        #fragment_indicies=range(i0,i0+FRAGSIZE) # put current fragment indicies in list
        last_pair_count = 0
        indicies= [] #list(fragment_indices) # copy current fragment indicies in list
        TM_temp =0.0
        # for TM_align it makes sense to make TOL larger since we want it to keep searching longer.
        t = 0.0  #################  nayve we should start this at 1???/
        while t<TOL:  # this goes one unit over TOL.  should we fix that?
          #  t+=UNIT  # e.g. one angstrom  # maybe I don't need this at all????  better without it?
            t+=0.1 #0.25*UNIT
            t2 = t*t # squared radius
            # this embeds a logic error in mammoth!  need to fix that!The percentage of structural similarity (PSI) is defined as the number of aligned amino acid pairs with Cα atoms that are closer in space then 4 Å after
            # problem happens if NO residues get included on first pass
            # then the t gets raised too much!

            min_d2 = (TOL+UNIT)*(TOL+UNIT)

            # have to us a fresh copy because we will be saving this
            #indicies= [] #list(fragment_indices) # copy current fragment indicies in list

            ### thought: woould it make sense to add residues to index not based on RMS but on TM-score?  Maybe not just the closest ones
            ###   but ones where the derivative of TM score is largest, dialing those in would actually improve the
            ###   TM score the most. Howver those are doing to be at the raidus d2= TM_d02  not d2~0

            #### thought:  apply weight matrix to the RMSD matrix composiiton so that it up weights the high TM score ones
            ####   alternative thought:  up weight the the ones that have the  highest derivative.

            ### insight:  weighting not only affects the rotation matrix but it also affects the translations vector.
            ###  thought:  maybe you do the first run maximizing TM score then the second run starts there and
            ####           tries maximizing according to the derivative weighting?  Maybe a linear combination of the two?



            for j0 in  range(N):
                # check if we should include this atom pair
                if j0 not in indicies:
                    deviations =(p0aligned[j0]-p1aligned[j0])**2
                    d2 = np.sum(deviations)  # MSD for atom pair so it's divide by 1 not L_align
                    if d2<t2 or (0<=j0-i0<FRAGSIZE): # could use short circuit or
                        indicies.append(j0)  # change this to faster method

                    else: # do not update min_d2 this for fragment itself
                        min_d2 = min(min_d2,d2) # track closest distance
                        # note this min ignores the ones in the aligned fragment.
            # did we add any new atomns in last iteration?
            L_indicies = len(indicies)

            # here we have to be careful--- as long as indicies is strictly accumaltive then we can be sure
            # that nothing in the alignemnt was added unless  L_indicies > last_pair_count
            # but if that is NOT true, and we can remove underperforming residues as the alignment focus
            # moves to better TM scores but fewer residues then we may not trigger this
            # since inside this we trigger the re-alignment then the evolution will stop in it's tracks!
            if  L_indicies > last_pair_count and  L_indicies >3:  #### may want to chack that have atleast 3 atoms of kabsch will barf.
                last_pair_count= L_indicies
                p0aligned,p1aligned,G=kabsch_template_alignment(p0,p1,p0[indicies],p1[indicies])

                deviation2 = np.sum((p0aligned-p1aligned)**2,axis=1)  # note the sum over coords!!  # this is over the full alignment.
                alignedRMS = np.sqrt(np.mean(deviation2))  # (assumes no replicates!)


                if (L_indicies>maxsub_most and alignedRMS<=RMSTOL) or \
                    (L_indicies==maxsub_most and alignedRMS<maxsub_alignedRMS):
                    maxsub_most = L_indicies
                    maxsub_alignedRMS= alignedRMS
                    maxsub_alignment= np.array(indicies)
                    maxsub_rotation = G


                #TM SCORED
                maxsub_TM_score_temp = np.sum(1.0/(1.0+deviation2/TM_d02 ))/L_min  # can use L_target instead of L_min for one-to-many

               # if (maxsub_TM_most > 3*FRAGSIZE):
                if ((L_indicies>  longest_TM_most) and  ( maxsub_TM_score_temp> 0.97*longest_TM_score_best)) or\
                   ((L_indicies<  longest_TM_most) and  ( maxsub_TM_score_temp> 1.02*longest_TM_score_best)) or\
                   ((L_indicies== longest_TM_most) and  ( maxsub_TM_score_temp>      longest_TM_score_best)):
                    longest_TM_score_best = maxsub_TM_score_temp
                    longest_TM_best_rotation = G # Do we need a copy of this? this si worthless without offset!!!!!#####
                    longest_TM_best_seed_alignment=np.array(indicies) # copy)
                    longest_TM_alignedRMS = alignedRMS
                    longest_TM_most = L_indicies

                if maxsub_TM_score_temp> maxsub_TM_score_best:

                    maxsub_TM_score_best = maxsub_TM_score_temp
                    maxsub_TM_best_rotation = G # Do we need a copy of this? this si worthless without offset!!!!!#####
                    maxsub_TM_best_seed_alignment=np.array(indicies) # copy)
                    maxsub_TM_alignedRMS = alignedRMS
                    maxsub_TM_most = L_indicies

                # be careful with the logic of which array to rotate
            else: # nothing was close enough at this tolerance
                t =np.sqrt(min_d2)# Will also add to this 0.1*UNIT above




        L_indicies = len(indicies)

        ### could the following maybe go wrong is the above isn't called?wrong.  it uses the offset for a different set sizer!!!!!!  ####$$$$$$$$$$$???????


        alignedRMS = np.sqrt(np.sum((p0aligned[indicies]-p1aligned[indicies])**2)/L_indicies)

        #### LOGIC ERROR--- WE NEED TO BE TRACKING THE maxsub score at every stage because it is
        ## # possible tha one gets to this point and alignedRMS > RMSTOL, so we won't record it
        #####  but there might have been an intermediate stage where there was a good one less than RMSTOL but
        #####  but with fewer aligned residues.

        #### is this an error in mammoth too?


    # now we can pick the best trade between trying for longer and trying for the best Tm
    if longest_TM_most> maxsub_TM_most and longest_TM_score_best > 0.97*maxsub_TM_score_best:
        print ("length trade off {} residues @ Tm{} verus {} residues @ Tm{}".format(maxsub_TM_most,maxsub_TM_score_best,longest_TM_most,longest_TM_score_best))
        maxsub_TM_score_best =          longest_TM_score_best
        maxsub_TM_best_rotation =       longest_TM_best_rotation
        maxsub_TM_best_seed_alignment=  longest_TM_best_seed_alignment
        maxsub_TM_alignedRMS =          longest_TM_alignedRMS
        maxsub_TM_most =                longest_TM_most

    # and as a final check we just do the full enchilada
    if True:  # since were going to score_metric anyhow, we can skip this
        p0aligned,p1aligned,G=kabsch_template_alignment(p0,p1,p0,p1)
      #  R,w,d,offset0,offset1 = kabsch(frag0,frag1) # just align this fragment

       # p1aligned = np.matmul(p1-offset1,R.T) # apply alignment.

        # The initial alginment  might be okay, so lets record it
        deviation2 = np.sum((p0aligned-p1aligned)**2,axis=1) # only sum over x,y,z but not over points.
        DEBUG2_TM_score_temp = np.sum(1.0/(1.0+deviation2/TM_d02 ))/L_min # can use L_target instead of L_min for one-to-many
        #    print ("iitial fragment msd ",np.sum(deviation2)/FRAGSIZE)
        #    print (align_index[:,i0:i0+FRAGSIZE])
        raw_rmsd_temp= np.sqrt(np.mean(deviation2))


#         if  maxsub_TM_score_best < raw_TM_score_temp:
#             maxsub_TM_score_best = raw_TM_score_temp
#             maxsub_TM_best_rotation = G
#             maxsub_TM_best_seed_alignment = range(L_aligned)
#             maxsub_TM_alignedRMS=raw_rmsd_temp
    # this is debug check
        p0aligned,p1aligned,G=kabsch_template_alignment(p0,p1,p0[maxsub_TM_best_seed_alignment],p1[maxsub_TM_best_seed_alignment])
        deviation2 = np.sum((p0aligned-p1aligned)**2,axis=1)  # note the sum over coords!!  # this is over the full alignment.
        alignedRMS = np.sqrt(np.mean(deviation2))  # (assumes no replicates!)
        check_TM_SCORE =  np.sum(1.0/(1.0+deviation2/TM_d02 ))/L_min
        print( "debug Tm scores \nrecomputed maxsub Tm{} =? best {} >? full_protein{}\nlen orietned {} full {}".format(check_TM_SCORE ,maxsub_TM_score_best, DEBUG2_TM_score_temp,len(maxsub_TM_best_seed_alignment),L_aligned))
        if   np.abs(1-maxsub_TM_score_best/ check_TM_SCORE) >0.05:
            print( "ERROR!!!!!!  these should agree {}  {}".format(maxsub_TM_score_best,check_TM_SCORE))
        if  maxsub_TM_score_best/DEBUG2_TM_score_temp > 1.05:
            print("WHOA:  maxsub found better answer residues {} @ Tm {}  better than {} @ Tm {}".format(maxsub_TM_most,maxsub_TM_score_best,L_aligned,DEBUG2_TM_score_temp))
        if   maxsub_TM_score_best/DEBUG2_TM_score_temp <0.94:
            print( "WA!!!!!! maxsub is worse than the full length alignment {} {}".format(maxsub_TM_most,maxsub_TM_score_best,L_aligned,DEBUG2_TM_score_temp))
        print("metrix")
        print(standard_metrics( master_p0,  master_p1,align_index,indicies=maxsub_TM_best_seed_alignment,seq0=None,seq1=None,d0=4.0, UNIT=1.0) )
        print(standard_metrics( master_p0,  master_p1,align_index,indicies=None,seq0=None,seq1=None,d0=4.0, UNIT=1.0) )
    return (MAXSUB_TM(maxsub_TM_score_best,maxsub_TM_best_rotation,maxsub_TM_best_seed_alignment,maxsub_TM_alignedRMS),
            MAXSUB_TM (raw_TM_score_best,raw_TM_best_rotation,raw_TM_best_seed_alignment,raw_TM_alignedRMS),
            MAXSUB_TM(maxsub_most,maxsub_rotation,maxsub_alignment,maxsub_alignedRMS))


Metrics = namedtuple('Metrics',['TM','PSI','aPSI','oPSI','rPSI','cRMS','aRMS','oRMS','aSeq_ident','oSeq_ident','cSeq_Ident','L_min','L_aligned','L_orientable','L_PSI'])


def standard_metrics(master_p0, master_p1,align_index,indicies,seq0=None,seq1=None,d0=4.0, UNIT=1.0):
    '''The percentage of structural similarity (PSI) is defined as the number of aligned
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

    TM_d0 = 1.24*(L_min-15)**0.333333 -1.8  # for the TM score
    TM_d02 = TM_d0**2

    p0 = master_p0[align_index[0]]
    p1 = master_p1[align_index[1]]




    p0aligned, p1aligned,G= kabsch_template_alignment(p0,p1,p0[indicies],p1[indicies])
    # rotation aligns just the selected atoms

    deviation2 = np.sum((p0aligned-p1aligned)**2,axis=1)  # this is over the full alignment.


    TM_score = np.sum(1.0/(1.0+deviation2/TM_d02 ))/L_min


    RMS = np.sqrt(np.sum(deviation2)/L_aligned)             #RMS over all
    oRMS = np.sqrt(np.sum(deviation2[indicies])/ L_orientable)  # RMS over subset of resides contributing to orientation

    PSI_mask =  np.sqrt(deviation2)<(4.0*UNIT)
    L_PSI = np.sum(PSI_mask)
    PSI= L_PSI/L_min


    if L_PSI>2: # RMS doesn't mean much otherwise
        # RMS over resides contributing to PSI
        cRMS = np.sqrt(np.sum(deviation2[PSI_mask])/L_PSI)  # or should be re-align this?  could become tailchasing?
    else:
        cRMS = np.NaN  # fiction.
    if seq0 is not None and seq1 is not None:
        seq_aligned = np.array([ [seq0[i],seq1[j]] for i,j in align_index.T ])
        aSeq_ident= np.sum( seq_aligned[:,0]==seq_aligned[:,1])

        aSeq_ident/=L_aligned

        oSeq_ident=np.sum( seq_aligned[indicies,0]==seq_aligned[indicies,1])

        oSeq_ident/=L_orientable

        cSeq_ident=np.sum( seq_aligned[PSI_mask,0]==seq_aligned[PSI_mask,1])

        cSeq_ident/=L_PSI
    else:
        aSeq_ident=oSeq_ident=cSeq_ident=0

    c=0 # run length tracker
    aPSI=0 # running sum of long runs with 4 or more residues in a row without gaps
    for i in range( L_aligned):
        c+=1
        if i+1==L_aligned or np.any((align_index[:,i+1]-align_index[:,i])>1) :  # a gap in either protein's alignment
           if c>3: aPSI+=c  # should I be checking for 3 or 4 here?
           c=0
    aPSI = aPSI/L_min

    oPSI=0 # running sum of long runs with 4 or more residues in a row without gaps
    for i in range( L_orientable):
        c+=1
        if i+1== L_orientable or np.any((align_index[:,indicies[i+1]]-align_index[:,indicies[i]])>1) :  # a gap in either protein's alignment
           if c>3: oPSI+=c  # should I be checking for 3 or 4 here?
           c=0
    oPSI = oPSI/L_min

    rPSI=0 # running sum of long runs with 4 or more residues in a row without gaps
    for i in range( L_PSI):
        c+=1
        if i+1==L_PSI or np.any((align_index[:,PSI_mask][:,i+1]-align_index[:,PSI_mask][:,i])>1) :  # a gap in either protein's alignment
           if c>3: rPSI+=c  # should I be checking for 3 or 4 here?
           c=0
    rPSI = rPSI/L_min

    return Metrics(TM_score,PSI,aPSI,oPSI,rPSI,cRMS,RMS,oRMS,aSeq_ident,oSeq_ident,cSeq_ident,L_min,L_aligned,L_orientable,L_PSI)


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


def process_alignment(alignment, seq0=None, seq1=None, pdb0=None, pdb1=None, transpose=True):
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
    _, fpnts0 = readPDB(pdb0)
    _, fpnts1 = readPDB(pdb1)  
    if transpose:
        fpnts0, fpnts1 = fpnts1, fpnts0
        seq0, seq1 = seq1, seq0

      
    a1 = parseAlingmentString(alignment)
    if seq0 is None or seq1 is None:
        seq0 = fpnts0.seq
        seq1 = fpnts1.seq
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
                          seq0=fpnts0.seq,
                          seq1=fpnts1.seq,
                          d0=4.0, UNIT=1.0)
    return sm
