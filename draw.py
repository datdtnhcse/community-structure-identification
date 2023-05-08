import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from operator import mul
from functools import *

def getEdge(G,G_part):
    edge_G = G.edges()
    edge_G_part = G_part.edges()
    list_edge = []
    for edge in edge_G:
        if edge in edge_G_part:
            continue
        list_edge.append(edge)
    return list_edge

def plotPartition(G,true_parttion):
    pos = nx.spring_layout(G, k=0.1, iterations=50, scale=1.3)
    colors = ["violet","orange","cyan","red","blue","green","yellow","indigo","pink"]
    for i,com in enumerate(true_parttion):
        nodes = list(com)
        nx.draw_networkx_nodes(G,pos,nodelist=nodes,node_size=500, node_color=colors[i % len(colors)])
    nx.draw_networkx_edges(G,pos,edgelist=G.edges(data=True), width=2,alpha=1,edge_color='k')
    nx.draw_networkx_labels(G,pos,font_size=12,font_family='sans-serif')

    plt.axis('off')
    plt.savefig('img.png')

def plot_graph(G, G_part):
    removed_edges = getEdge(G, G_part)
    exist_edges = G_part.edges(data=True)
    pos = nx.spring_layout(G, k=0.1, iterations=50, scale=1.3)
    
    # nodes
    coms = nx.connected_components(G_part)
    colors = ["violet","black","orange","cyan","red","blue","green","yellow","indigo","pink"]
    for i,com in enumerate(coms):
        nodes = list(com) 
        # np.random.seed( len(nodes)*sum(nodes)*reduce(mul, nodes, 1)*min(nodes)*max(nodes) )
        # colors = np.random.rand(4 if len(nodes)<=4 else len(nodes))
        nx.draw_networkx_nodes(G,pos,nodelist=nodes,node_size=500, node_color=colors[i % len(colors)])

    # edges
    nx.draw_networkx_edges(G,pos,edgelist=exist_edges, width=2,alpha=1,edge_color='k')
    nx.draw_networkx_edges(G,pos,edgelist=removed_edges, width=2, edge_color='k')   #, style='dashed')

    # labels
    nx.draw_networkx_labels(G,pos,font_size=12,font_family='sans-serif')

    plt.axis('off')
    plt.savefig('img.png')