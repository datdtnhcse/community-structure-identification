import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

import networkx as nx
from networkx.algorithms.community import louvain_communities
from util import *
from louvain_method import *
from sklearn.metrics.cluster import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score

adj_matrix = get_data(20)
G = nx.from_numpy_array(adj_matrix)

communities = louvain_communities(G, seed = 123)
true_partition, frame = louvain_algorithm(adj_matrix)
communities = [list(ele) for ele in communities]
true_partition = [list(ele) for ele in true_partition]



print(true_partition)
print(communities)