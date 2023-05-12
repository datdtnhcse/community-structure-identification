import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

import numpy as np
from collections import defaultdict
import networkx as nx
from util import *
from draw import *
from numpy.linalg import lstsq
# from als import *
# from mu import *
# from anls_as import *
# from sgd import *

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

def getCluster(W):
    sorted_W = W[:, np.argsort(W.sum(axis = 0))]
    cluster = defaultdict(lambda : [])
    # print(W.shape)
    for i in range(np.size(W,0)):
        idx = (-sorted_W[i,:]).argsort()[0]
        cluster[idx].append(i)
    lst_cluster = []
    for _,val in cluster.items():
        lst_cluster.append(set(val))
    return lst_cluster

def algoNMF(adjency_matrix, num_cluster, num_iter,algo = 'anls_as'):
    if algo == 'anls_as':
        W, H = alns_as(adjency_matrix,num_cluster,num_iter)
    elif algo == 'als':
        W, H = als(adjency_matrix,num_cluster,num_iter)
    elif algo == 'mu':
        W, H = mu(adjency_matrix,num_cluster,num_iter = num_iter)
    elif algo == 'sgd':
        W,H,cost = sgd(adjency_matrix,num_cluster,num_iter)
    # print(W)
    return getCluster(W)
    
# adj_matrix = getAdjMatrix("dataset/edge.txt",500)
# adj_matrix = get_data(50)
# G = nx.from_numpy_array(adj_matrix)
# true_partition = algoNMF(adj_matrix,8,10000,'sgd')
# # plotPartition(G,true_partition)
# print(true_partition)