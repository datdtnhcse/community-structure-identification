import numpy as np

#paper ref: https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf

def mu(adjency_matrix, num_cluster, num_iter,delta = 0.01, init_W = None, init_H = None, print_enabled = True):  
    if print_enabled:
        print('---------------------------------------------------------------------')
        print('Frobenius norm ||A - WH||_F')
        print('')

    if init_W is None:
        W = np.random.rand(np.size(adjency_matrix, 0), num_cluster)
    else: W = init_W
    if init_H is None:
        H = np.random.rand(num_cluster, np.size(adjency_matrix, 1))
    else: H = init_H
    
    for n in range(num_iter):
        #update H
        W_TA = W.T @ adjency_matrix #size ((k,n) * (n,n) => (k,n))
        W_TWH = W.T @ W @ H + delta #size ((k,n) * (n,k) * (k,n) = (k,n))
        for i in range(np.size(H, 0)):
            for j in range(np.size(H, 1)):
                H[i, j] = H[i, j] * W_TA[i, j] / W_TWH[i, j]
        
        #update W
        AH_T = adjency_matrix @ H.T #((n,n) * (n,k) => (n,k))
        WHH_T = W @ H @ H.T + delta#((n,k) * (k,n) * (n,k)  => (n,k))
        for i in range(np.size(W,0)):
            for j in range(np.size(W,1)):
                W[i, j] = W[i, j] * AH_T[i, j] / WHH_T[i, j]
                
        if print_enabled:
            frob_norm = np.linalg.norm(adjency_matrix - W @ H, 'fro')
            print("iteration " + str(n + 1) + ": " + str(frob_norm))
    
    return W, H