import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

import numpy as np
from collections import defaultdict
import networkx as nx
from util import *
from draw import *
from als import *
from mu import *
from anls_as import *
from sgd import *

def getCluster(W):
    sorted_W = W[:, np.argsort(W.sum(axis = 0))]
    cluster = defaultdict(lambda : [])
    print(W.shape)
    for i in range(np.size(W,0)):
        idx = (-sorted_W[i,:]).argsort()[0]
        cluster[idx].append(i)
    return cluster

def algoNMF(adjency_matrix, num_cluster, num_iter,algo = 'anls_as'):
    if algo == 'anls_as':
        W, H = alns_as(adjency_matrix,num_cluster,num_iter)
    elif algo == 'als':
        W, H = als(adjency_matrix,num_cluster,num_iter)
    elif algo == 'mu':
        W, H = mu(adjency_matrix,num_cluster,num_iter = num_iter)
    elif algo == 'sgd':
        W,H,cost = sgd(adjency_matrix,num_cluster,num_iter)
    print(W)
    return getCluster(W)
    
adj_matrix = get_data(10)
G = nx.from_numpy_array(adj_matrix)
true_partition = algoNMF(adj_matrix,4,32,'sgd')
plotPartition(G,true_partition)
print(true_partition)