import numpy as np

def sgd(adjency_matrix, num_cluster, num_iter, lr = 0.01, init_W = None, init_H = None, print_enabled = True):  
    cost = []
    if init_W is None:
        W = np.random.rand(np.size(adjency_matrix, 0), num_cluster)
    else: W = init_W
    if init_H is None:
        H = np.random.rand(num_cluster, np.size(adjency_matrix, 1))
    else: H = init_H
    
    for i in range(num_iter):
        grad_W = W @ H @ H.T - adjency_matrix @ H.T
        grad_H = W.T @ W @ H - W.T @ adjency_matrix
        W = W - lr * grad_W
        H = H - lr * grad_H
        W[W < 0] = 0
        H[H < 0] = 0
        cost.append(np.abs(adjency_matrix - W@H).sum()) #MAE
        if print_enabled:
            frob_norm = np.linalg.norm(adjency_matrix - W @ H, 'fro')
            print("iteration " + str(i + 1) + ": " + str(frob_norm))
    
    return W, H, cost