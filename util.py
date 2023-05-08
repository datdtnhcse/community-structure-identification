import numpy as np
import pandas as pd
from typing import Callable
from itertools import product, combinations

np.random.seed(4)

def get_data(k):
        # np.random.seed(10)
    data = np.zeros((k,k))
    for i in range(len(data)):
        for j in range(len(data)):
            data[i,j] = np.random.choice([0,1], p=[0.6, 0.4])
    data = data + np.multiply(data.T, data.T > data) - np.multiply(data, data.T > data)
    for i in range(len(data)):
        data[i, i] = 0
        # data[i,k] = 0
        # data[k,i] = 0
    return data

def intercommunity_matrix(adj_matrix, communities, aggr: Callable = sum):
    num_nodes = len(communities)
    intercomm_adj_matrix = np.zeros((num_nodes, num_nodes))
    for i, src_comm in enumerate(communities):
        for j, targ_comm in enumerate(communities):
            if j > i:
                break
            edge_weights = []
        for u, v in product(src_comm, targ_comm):
            edge_weights.append(adj_matrix[u, v])
        edge_weight = aggr(edge_weights)
        intercomm_adj_matrix[i, j] = edge_weight
        intercomm_adj_matrix[j, i] = edge_weight
    
    return intercommunity_matrix()
        
def laplacian_matrix(adj_matrix : np.ndarray):
    diagonal = adj_matrix.sum(axis=1)
    D = np.diag(diagonal)
    L = D - adj_matrix

    return L

def modularity_matrix(adj_matrix : np.ndarray):
    k_i = np.expand_dims(adj_matrix.sum(axis=1), axis=1)
    # print(k_i)
    k_j = k_i.T
    norm = 1 / k_i.sum()
    K = norm * np.matmul(k_i, k_j)

    return (adj_matrix - K)

def modularity(mod_matrix : np.ndarray, communities : list):
    C = np.zeros_like(mod_matrix)
    for community in communities:
        for i, j in combinations(community, 2):
            C[i, j] = 1.0
            C[j, i] = 1.0
    # print(C)
    return np.tril(np.multiply(mod_matrix, C), 0).sum()
        
if __name__ == "__main__":
    data = get_data(4)
    print(data)
    # print(modularity_matrix(data))
    # print(betweennes_matrix(data))
    # lap_matrix = laplacian_matrix(data)
    # print(lap_matrix)

    # eigenvalues, eigenvectors = np.linalg.eig(lap_matrix)
    # print(eigenvalues)