import networkx as nx
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)

from util import *

def get_max_betweenness_edges(betweenness):
    max_betweenness_edges = []
    max_betweenness = max(betweenness.items(), key=lambda x: x[1])
    for (k, v) in betweenness.items():
        if v == max_betweenness[1]:
            max_betweenness_edges.append(k)
    return max_betweenness_edges

def build_level( G, root):
    """ 
    Build level for graph

    input:
    - G: networkx graph
    - root: root node

    output:
    - levels: nodes in each level
    - predecessors: predecessors for each node
    - successors: successors for each node
 
    """
    levels = {}
    predecessors = {}
    successors = {}

    cur_level_nodes = [root]    # initialize start point
    nodes = []  # store nodes that have been accessed
    level_idx = 0       # track level index
    while cur_level_nodes:  # if have nodes for a level, continue process
        nodes.extend(cur_level_nodes)   # add nodes that are inside new level into nodes list
        levels.setdefault(level_idx, cur_level_nodes)   # set nodes for current level
        next_level_nodes = []   # prepare nodes for next level

        # find node in next level
        for node in cur_level_nodes:
            nei_nodes = G.neighbors(node)   # all neighbors for the node in current level
            
            # find neighbor nodes in the next level
            for nei_node in nei_nodes:
                if nei_node not in nodes:   # nodes in the next level must not be accessed
                    predecessors.setdefault(nei_node, [])   # initialize predecessors dictionary, use a list to store all predecessors
                    predecessors[nei_node].append(node) 
                    successors.setdefault(node, [])     # initialize successors dictionary, use a list to store all successors
                    successors[node].append(nei_node)

                    if nei_node not in next_level_nodes:    # avoid add same node twice
                        next_level_nodes.append(nei_node)
        cur_level_nodes = next_level_nodes
        level_idx += 1
    return levels, predecessors, successors

def calculate_credits( G, levels, predecessors, successors, nodes_nsp):
    """
    Calculate credits for nodes and edges

    """
    nodes_credit = {}
    edges_credit = {}

    # loop, from bottom to top, not including the zero level
    for lvl_idx in range(len(levels)-1, 0, -1):
        lvl_nodes = levels[lvl_idx]     # get nodes in the level

        # calculate for each node in current level
        for lvl_node in lvl_nodes:
            nodes_credit.setdefault(lvl_node, 1.)   # set default credit for the node, 1
            if lvl_node in successors.keys():        # if it is not a leaf node
                # Each node that is not a leaf gets credit = 1 + sum of credits of the DAG edges from that node to level below
                for successor in successors[lvl_node]:
                    nodes_credit[lvl_node] += edges_credit[(successor, lvl_node)]

            node_predecessors = predecessors[lvl_node]  #  get predecessors of the node in current level
            total_nodes_nsp = .0    # total number of shortest paths for predecessors of the node in current level
            
            # sum up for total_nodes_nsp
            for predecessor in node_predecessors:
                total_nodes_nsp += nodes_nsp[predecessor]

            # again, calculate for the weight of each predecessor, and assign credit for the responding edge
            for predecessor in node_predecessors:
                predecessor_weight = nodes_nsp[predecessor]/total_nodes_nsp     # calculate weight of predecssor
                edges_credit.setdefault((lvl_node, predecessor), nodes_credit[lvl_node]*predecessor_weight)         # bottom-up edge
    return nodes_credit, edges_credit


def my_betweenness_calculation( G, normalized=False):
    """
    Main Bonus Function to calculation betweenness

    """
    graph_nodes = G.nodes()
    edge_contributions = {}
    # components = list(nx.connected_components(G)) # connected components for current graph
    components = getComponent(G)
    # calculate for each node
    for node in graph_nodes:
        component = None    # the community current node belongs to
        for com in components: 
            if node in com:
                component = com
        nodes_nsp = {}  # number of shorest paths
        node_levels, predecessors, successors = build_level(G, node)   # build levels for calculation

        # calculate shortest paths for each node (including current node)
        for other_node in component:
            shortest_paths = nx.all_shortest_paths(G, source=node,target=other_node)
            nodes_nsp[other_node] = len(list(shortest_paths))

        # calculate credits for nodes and edges (Only use "edges_credit" actually)
        nodes_credit, edges_credit = calculate_credits(G, node_levels, predecessors, successors, nodes_nsp)

        # sort tuple (key value of edges_credit), and sum up for edge_contributions
        for (k, v) in edges_credit.items():
            k = sorted(k, reverse=False)
            edge_contributions_key = (k[0], k[1])
            edge_contributions.setdefault(edge_contributions_key, 0)
            edge_contributions[edge_contributions_key] += v
       
    # divide by 2 to get true betweenness
    for (k, v) in edge_contributions.items():
        edge_contributions[k] = v/2

    # normalize
    if normalized:
        max_edge_contribution = max(edge_contributions.values())
        for (k, v) in edge_contributions.items():
            edge_contributions[k] = v/max_edge_contribution
    return edge_contributions