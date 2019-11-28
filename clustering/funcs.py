import pandas as pd
import numpy as np
import networkx as nx

__all__ = ('k_cluster', )

def k_cluster(G, k):
    """returns k clusters from the graph G via the method described in class:

    randomly selects a starting vertex and then places k vertices at points furthest away from all cluster centers.
    then assigns all remaining points to the cluster center closest to each point.

    ideas for improvement:
    1. don't assign all points to clusters, choose some radius to which, if a point is that far away from a cluster center,
        then it is assigned to that cluster otherwise none.
    2. some metric to guess how many clusters, look at degree of clustering possibly
    """

    used = set()
    startnode = np.random.choice(G.nodes, 1)[0]
    used.update({startnode})
    lens = nx.shortest_path_length(G, startnode, weight = 'weight')

    nused = 1
    for j in range(k-1):
        mind = dict()
        for node in G.nodes:
            if node in used:
                continue
            #for each node, find the min distance to the used nodes. then take the max of this put this in a new cluster
            ltoused = [nx.dijkstra_path_length(G, node, nodex) for nodex in used]
            mind.update({node:min(ltoused)})
        #key of the furtherest away i.e. the node furthest away
        newnode = max(mind, key=mind.get)
        used.update({newnode})
        nused+=1

    out = dict()
    for u in used:
        out.update({u:[]})
    #now put all nodes in appropriate cluster:
    for node in G.nodes:
        l=nx.shortest_path_length(G, node, weight = 'weight')
        d = np.inf
        for u in used:
            le = l[u]
            if le < d:
                d = le
                clustercenter = u
        out[clustercenter].append(node)

    return out
