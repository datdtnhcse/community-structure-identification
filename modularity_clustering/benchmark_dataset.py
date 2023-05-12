import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

import warnings
warnings.filterwarnings('ignore')
import networkx as nx
from surprise import NMF as nmf_als
from networkx.algorithms.community import louvain_communities
from util import *
from louvain_method import *
from girvan_newan import *
# from sklearn.metrics.cluster import adjusted_rand_score, fowlkes_mallows_score, normalized_mutual_info_score
from benchmark_util import *
import nmf.main as nmf_util
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

#if use example data to get insight in result use
adj_matrix = get_data(100)

#benchmark for network-eletric
# adj_matrix = getAdjMatrix('input.txt')

#uncomment to run input if use multigraph
# adj_matrix = generate_adjacency_matrix_from_multigraph("multigraph.txt")

def convert_to_dictLabel(dict_test_util):
    dict_res = {}
    for x,ele_lst in dict_test_util.items():
        for ele in ele_lst:
            dict_res[ele] = x
    return dict_res

def elbow_num_component(adj_matrix):
    reconstruction_errors = []
    n = len(adj_matrix)
    print(n)
    lst_candidate = [4,8,16,32,64,128,256]
    n_components = []
    for ele in lst_candidate:
        if n >= ele:
            n_components.append(ele)
    print(n_components)
    for n in n_components:
        nmf = NMF(n_components=n, init='random', random_state=0, max_iter = 1000)
        nmf.fit(adj_matrix)
        reconstruction_error = nmf.reconstruction_err_
        reconstruction_errors.append(reconstruction_error)
    # plt.plot(n_components, reconstruction_errors, marker='o')
    # plt.xlabel('Number of Components')
    # plt.ylabel('Reconstruction Error')
    # plt.title('Elbow Method for NMF')
    diff = np.diff(reconstruction_errors)
    elbow_index = np.argmax(diff) + 1
    # plt.axvline(x=elbow_index, color='r', linestyle='--', label='Elbow')

    # plt.xticks(n_components)
    # plt.show()
    return n_components[elbow_index]

def method_to_measure(adj_matrix,method = 'louvain',n_component = None,name = None):
    G = nx.from_numpy_array(adj_matrix)
    communities = None
    true_partition = None
    if n_component is None:
        num_com = elbow_num_component(adj_matrix)
        print(num_com)
    else: num_com = n_component
    if method == 'louvain':
        print('method: louvain')
        communities = louvain_communities(G, seed = 123)
        true_partition, frame = louvain_algorithm(adj_matrix)
    if method == 'girvan_newan':
        print('method: girvan_newan')
        communities = nx.algorithms.community.girvan_newman(G)
        lst_communities = []
        for partition in next(communities):
            lst_communities.append(partition)
        communities = lst_communities
        G_part, true_partition, _ = girvan_newan(adj_matrix, len(communities))    
    if method == 'nmf_sgd':
        print('method: nmf_sgd')
        true_partition = nmf_util.algoNMF(adj_matrix,num_com,10000,'sgd')
        skNMF = NMF(n_components=num_com, init='random', random_state=28, max_iter=10000, alpha=0.01)
        W1 = skNMF.fit_transform(adj_matrix)
        communities = nmf_util.getCluster(W1)
    if method == 'nmf_mu':
        print('method: nmf_mu')
        true_partition = nmf_util.algoNMF(adj_matrix,num_com,10000,'mu')
        nmf = NMF(n_components=num_com, init='random', random_state=0, solver='mu',max_iter=10000)
        W1 = nmf.fit_transform(adj_matrix)  # Ma trận W
        communities = nmf_util.getCluster(W1)
    # if method == 'nmf_als':
    #     model = nmf_als(n_factors=num_com, biased=False, random_state=0,n_epochs=10000)
    #     model.fit(adj_matrix)
    #     W = model.qi
    #     true_partition = nmf_util.algoNMF(adj_matrix,num_com,10000,'als')
    #     communities = nmf_util.getCluster(W)
    
    communities_ = {i + 1 : list(ele) for i,ele in enumerate(list(communities))}
    true_partition_ = {i + 1 : ele if list(ele) is not list else ele for i,ele in enumerate(true_partition)}
    
    dict_util = convert_to_dictLabel(true_partition_) 
    dict_algo = convert_to_dictLabel(communities_)
    list_label_algo = [dict_algo[ele] for ele in range(len(adj_matrix))]
    list_util_algo = [dict_util[ele] for ele in range(len(adj_matrix))]

    nmi = normalized_mutual_information(list_label_algo,list_util_algo)
    ari = adjusted_rand_index(list_label_algo,list_util_algo)
    fmi = fowlkes_mallows_index(list_label_algo,list_util_algo)
    
    df = pd.DataFrame([[nmi,ari,fmi]], columns=["NMI", "ARI", "FMI"])
    if name is None:
        df.index = [method]
    else: df.index = [name]
    return df
    
res = pd.DataFrame({})
res = pd.concat([res,method_to_measure(adj_matrix,'louvain')])
res = pd.concat([res,method_to_measure(adj_matrix,'girvan_newan')])
res = pd.concat([res,method_to_measure(adj_matrix,'nmf_sgd')])
res = pd.concat([res,method_to_measure(adj_matrix,'nmf_mu')])
# res = pd.concat([res,method_to_measure(adj_matrix,'nmf_als',8)])
print(res)