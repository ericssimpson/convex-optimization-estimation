import numpy as np

def add_bbox(A, b, xrange, yrange):
    A = np.vstack((A, [
        [-1,  0],
        [ 1,  0],
        [ 0, -1],
        [ 0,  1],
    ]))
    b = np.hstack((b, [-xrange[0], xrange[1], -yrange[0], yrange[1]]))
    return A, b
