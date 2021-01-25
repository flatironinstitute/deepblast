# state numberings
# x: insertion in protein x
# y: insertion in protein y
# m: match
# s: slip in both
x, m, y = 0, 1, 2

# Below are the tentative encodings for
# the affine gap alignment
m_, x_, y_, s_ = 0, 1, 2, 3

# indices for 3 state HMM
pos_mxy = [(-1, -1),
           (-1, 0),
           (0, -1)]

# indices for 4 state HMM with slip state
pos_mxys = [(-1, -1),
            (-1, 0),
            (0, -1),
            (-1, -1)]

pos_test = [(-1, -1),
            (-1, 0)]
