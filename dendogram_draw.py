import networkx as nx
from scipy.cluster.hierarchy import linkage
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
A = nx.to_numpy_array(G)
Z = linkage(np.asarray(A), 'ward')

plt.figure(figsize=(10, 5))
dn = dendrogram(Z)
plt.show()