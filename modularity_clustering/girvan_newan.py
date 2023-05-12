import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from util import *
import networkx as nx
from betweeness import *
from draw import *
import copy

# Function to prune edges based on betweenness centrality
def prune_edges(G):
    init_num_comps = nx.number_connected_components(G) 
    curr_num_comps = init_num_comps
    
    while curr_num_comps <= init_num_comps:
        # Calculate betweenness centrality for each edge
        bw_centralities = my_betweenness_calculation(G, True)
        
        # Get the edge(s) with maximum betweenness centrality
        max_bw_edge = get_max_betweenness_edges(bw_centralities)
        
        # Remove the edge(s) with maximum betweenness centrality
        for edge in max_bw_edge:
            G.remove_edge(*edge)
            
        # Update the number of connected components
        curr_num_comps = nx.number_connected_components(G)
        
    return G

# Function to convert partition history and modularity values into animation frames
def animation_data(A, P_history, Q_history):
    num_nodes = len(A)
    frames = []
    
    for P, Q in zip(P_history, Q_history):
        _P = [0 for _ in range(num_nodes)]
        
        # Assign a unique index to each node based on the partition it belongs to
        for index, partition in enumerate(P):
            for node in partition:
                _P[node] = index
                
        frames.append({"C" : _P, "Q" : Q})
        
    return frames

# Get the adjacency matrix (either from a file or a custom function)
# adj_matrix = get_data(20)
# adj_matrix = getAdjMatrix('input.txt')

# Girvan-Newman algorithm
def girvan_newan(adj_matrix, n = None):
    # Calculate the modularity matrix
    M = modularity_matrix(adj_matrix)
    
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    num_nodes = G.number_of_nodes()
    
    # Remove self-loops from the graph
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # Initial partition (all nodes in one community)
    best_P = getComponent(G)
    best_Q = modularity(M, best_P)
    best_G = G
    
    # Keep track of partition history and modularity values
    P_history = [best_P]
    Q_history = [best_Q]
    
    while True:
        last_P = P_history[-1]
        
        # Check termination conditions
        if not n and len(last_P) == num_nodes:
            return best_G, best_P, animation_data(adj_matrix, P_history, Q_history)
        elif n and len(last_P) == n:
            return best_G, last_P, animation_data(adj_matrix, P_history, Q_history)
        
        # Prune edges based on betweenness centrality
        G = prune_edges(G)
        
        # Get the new partition and calculate modularity
        P = getComponent(G)
        Q = modularity(M, P)
        
        # Update the best partition and modularity if necessary
        if Q >= best_Q:
            best_Q = Q
            best_P = P
            best_G = copy.deepcopy(G)
        
        # Record the partition and modularity values
        P_history.append(P)
        Q_history.append(Q)

if __name__ == "__main__":
    # Apply the Girvan-Newman algorithm to the adjacency matrix
    # adj_matrix = get_data(20)
    adj_matrix = getAdjMatrix('input.txt')
    G_part, component, _ = girvan_newan(adj_matrix)
    
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    
    # Plot the graph with community partitions
    plot_graph(G, G_part)
    
    # Print the best partition
    print("best_P: ", component)