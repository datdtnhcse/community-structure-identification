import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

import networkx as nx
from networkx.algorithms.community import louvain_communities
from util import *
from louvain_method import *
from girvan_newan import *
from sklearn.metrics.cluster import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score
from benchmark_util import *
import copy
# from nmf import *
adj_matrix = get_data(20)

def convert_to_dictLabel(dict_test_util):
    dict_res = {}
    for x,ele_lst in dict_test_util.items():
        for ele in ele_lst:
            dict_res[ele] = x
    return dict_res

def method_to_measure(adj_matrix,method = 'louvain'):
    G = nx.from_numpy_array(adj_matrix)
    communities = None
    true_partition = None
    if method == 'louvain':
        communities = louvain_communities(G, seed = 123)
        true_partition, frame = louvain_algorithm(adj_matrix)
    if method == 'girvan_newan':
        communities = nx.algorithms.community.girvan_newman(G)
        best_partition = None
        best_modularity = float('-inf')
        M = modularity_matrix(adj_matrix)
        G_copy = copy.deepcopy(G)
        for partition in next(communities):
            G_copy = prune_edges(G_copy)
            P = getComponent(G_copy)
            q = modularity(M, P)
            if q > best_modularity:
                best_partition = partition
                best_modularity = q
        communities = best_partition
        print("here", best_partition)
        G_part, true_partition, _ = girvan_newan(adj_matrix)    
    
    print(communities)
    communities_ = {i + 1 : list(ele) for i,ele in enumerate(list(communities))}
    print(communities_)
    true_partition_ = {i + 1 : ele if list(ele) is not list else ele for i,ele in enumerate(true_partition)}
    print("true: ", true_partition_)
    dict_util = convert_to_dictLabel(true_partition_) 
    dict_algo = convert_to_dictLabel(communities_)
    list_label_algo = [dict_algo[ele] for ele in range(len(adj_matrix))]
    list_util_algo = [dict_util[ele] for ele in range(len(adj_matrix))]

    nmi = normalized_mutual_information(list_label_algo,list_util_algo)
    ari = adjusted_rand_index(list_label_algo,list_util_algo)
    fmi = fowlkes_mallows_index(list_label_algo,list_util_algo)
    
    df = pd.DataFrame([[nmi,ari,fmi]], columns=["NMI", "ARI", "FMI"])
    df.index = [method]
    return df
    
res = pd.DataFrame({})
# res = pd.concat([res,method_to_measure(adj_matrix,method = 'louvain')])
res = pd.concat([res,method_to_measure(adj_matrix,method = 'girvan_newan')])
print(res)