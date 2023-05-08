import numpy as np
from numpy.linalg import lstsq

def as_vector(B, y, eps = 0.0001):
    #||Bg - y||_2.
    n = np.size(B,1)
    g = np.zeros(n)
    E = np.ones(n) #active set
    S = np.zeros(n) #passive set
    
    w = B.T @ (y - B @ g)
    wE = w * E
    t = np.argmax(wE)
    v = wE[t]
    
    while np.sum(E) > 0 and v > eps:
        # Update active/passive set indices
        E[t] = 0
        S[t] = 1
        
        Bs = B[:, S > 0]
        zsol = lstsq(Bs,y,rcond = -1)[0]
        zz = np.zeros(n)
        zz[S > 0] = zsol
        z = zz
        
        while np.min(z[S > 0]) <= 0:
            alpha = np.min((g / (g - z))[(S > 0) * (z <= 0)])
            g += alpha * (z - g)
            S[g == 0] = 0
            E[g == 0] = 1
            Bs = B[:, S > 0]
            zsol = lstsq(Bs, y)[0]
            zz = np.zeros(n)
            zz[S > 0] = zsol
            z = zz
        
        g = z
        w = B.T @ (y - B @ g)
        t = np.argmax(wE)
        v = wE[t]
    
    return g
    
def as_matrix(B, Y):
    return [np.array([as_vector(B, column) for column in Y.T]).T]

def alns_as(adjency_matrix, num_cluster, num_iter, init_W = None, init_H = None, print_enabled = True):  
    if init_W is None:
        W = np.random.rand(np.size(adjency_matrix, 0), num_cluster)
    else: W = init_W
    if init_H is None:
        H = np.random.rand(num_cluster, np.size(adjency_matrix, 1))
    else: H = init_H
    
    for i in range(num_iter):
        H = as_matrix(W, adjency_matrix)[0]
        W = as_matrix(H.T, adjency_matrix.T)[0].T
        
        if print_enabled:
            frob_norm = np.linalg.norm(adjency_matrix - W @ H, 'fro')
            print("iteration " + str(i + 1) + ": " + str(frob_norm))
    
    return W,H