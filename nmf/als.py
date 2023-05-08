import numpy as np
from numpy.linalg import lstsq

def als(adjency_matrix, num_cluster, num_iter, init_W = None, init_H = None, print_enabled = True):  
    if init_W is None:
        W = np.random.rand(np.size(adjency_matrix, 0), num_cluster)
    else: W = init_W
    if init_H is None:
        H = np.random.rand(num_cluster, np.size(adjency_matrix, 1))
    else: H = init_H
    
    for i in range(num_iter):
        # Solve the least squares problem: argmin_H ||WH - A||
        H = lstsq(W,adjency_matrix,rcond = - 1)[0]
        H[H < 0] = 0
        
        W = lstsq(H.T,adjency_matrix.T,rcond = -1)[0].T
        W[W < 0] = 0
        
        if print_enabled:
            frob_norm = np.linalg.norm(adjency_matrix - W @ H, 'fro')
            print("iteration " + str(i + 1) + ": " + str(frob_norm))
    
    return W,H