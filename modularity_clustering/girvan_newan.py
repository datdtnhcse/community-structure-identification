import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from util import *
import networkx as nx
from betweeness import *
from draw import *
import copy

def prune_edges(G):
    init_num_comps = nx.number_connected_components(G)
    curr_num_comps = init_num_comps
    
    while curr_num_comps <= init_num_comps:
        bw_centralities = my_betweenness_calculation(G, True)
        max_bw_edge = get_max_betweenness_edges(bw_centralities)
        
        for edge in max_bw_edge:
            G.remove_edge(*edge)
        curr_num_comps = nx.number_connected_components(G)
        
    return G
        
def animation_data(A, P_history, Q_history):
    num_nodes = len(A)
    frames = []
    for P,Q in zip(P_history,Q_history):
        _P = [0 for _ in range(num_nodes)]
        for index, partition in enumerate(P):
            for node in partition:
                _P[node] = index
        frames.append({"C" : _P, "Q" : Q})
    return frames
        
adj_matrix = get_data(20)

def girvan_newan(adj_matrix, n = None):
    M = modularity_matrix(adj_matrix)
    G = nx.from_numpy_array(adj_matrix)
    num_nodes = G.number_of_nodes()
    G.remove_edges_from(nx.selfloop_edges(G))
    
    best_P = list(nx.connected_components(G)) # Partition
    best_Q = modularity(M, best_P)
    best_G = G
    P_history = [best_P]
    Q_history = [best_Q]
    while True:
        last_P = P_history[-1]
        if not n and len(last_P) == num_nodes:
            return best_G,best_P,animation_data(adj_matrix, P_history, Q_history)
        elif n and len(last_P) == n:
            return best_G,last_P,animation_data(adj_matrix, P_history, Q_history)
        G = prune_edges(G)
        P = list(nx.connected_components(G))
        # print(P)
        Q = modularity(M,P)
        if Q >= best_Q:
            best_Q = Q
            best_P = P
            best_G = copy.deepcopy(G)
        P_history.append(P)
        Q_history.append(Q)
        
G_part,component,_ = girvan_newan(adj_matrix)
G = nx.from_numpy_array(adj_matrix)

plot_graph(G,G_part)

print("best_P: ",component)