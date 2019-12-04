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
    
    shortest_path_lens = dict(nx.all_pairs_dijkstra_path_length(G))
    
    
    #generate the k cluster centers
    for j in range(k-1):
        mind = dict()
        for node in G.nodes:
            if node in used:
                continue
            #for each node, find the min distance to the used nodes. then take the max of this put this in a new cluster
            ltoused = [shortest_path_lens[node][nodex] for nodex in used]
            mind.update({node:min(ltoused)})
        #key of the furtherest away i.e. the node furthest away
        newnode = max(mind, key=mind.get)
        used.update({newnode})

    out = dict()
    for u in used:
        out.update({u:[]})
    #now put all nodes in appropriate cluster:
    for node in G.nodes:
        l=shortest_path_lens[node]
        d = np.inf
        for u in used:
            le = l[u]
            if le < d:
                d = le
                clustercenter = u
        out[clustercenter].append(node)

    return out

def map_to_clusters(graph,cluster_centers,shortest_path_lengths):
    """
    returns a dict that maps each cluster center to a list of vertices which are closest
    to the given cluster center. Also returns a dict that maps each vertex to the distance 
    to the nearest cluster center
    Inputs:
    graph - input graph
    cluster_centers - list of vertices which are cluster centers
    shortest_path_lengths - a "2D dict" such that shortest_path_lengths[v1][v2] gives the length of 
        the shortest path from v1 to v2
    """
    mapping = dict()
    for node in cluster_centers:
        mapping[node] = []

    minimum_distance_to_centers = dict()
    for node in graph.nodes:
        distane_to_centers = [shortest_path_lengths[node][center] for center in cluster_centers]
        minimum_distance = min(distane_to_centers)
        minimum_distance_to_centers[node] = minimum_distance
        nearest_idx = distane_to_centers.index(minimum_distance)
        nearest_center = cluster_centers[nearest_idx]
        mapping[nearest_center].append(node)

    return mapping, minimum_distance_to_centers


def all_k_clusters(graph,num_clusters,all_pairs_distances):
    """
    returns the clustering for each cluster value from 1 to num_clusters
    """
    assert num_clusters <= graph.number_of_nodes(), "Max number of clusters is the number of vertices in graph"
    num_clusters_to_clustering = dict()
    cluster_centers = []

    startnode = np.random.choice(list(graph.nodes))
    cluster_centers.append(startnode)
    mapping, minimum_distance_to_centers = map_to_clusters(graph,cluster_centers,all_pairs_distances)

    num_clusters_to_clustering[1] = mapping

    for k in range(2,num_clusters+1):
        next_center = max(minimum_distance_to_centers, key = minimum_distance_to_centers.get)
        cluster_centers.append(next_center)
        mapping, minimum_distance_to_centers = map_to_clusters(graph,cluster_centers,all_pairs_distances)
        num_clusters_to_clustering[k] = mapping

    return num_clusters_to_clustering


def k_cluster_old(G, k):
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

def best_dropoff(G, subset):
    '''best_dropoff(nx.graph, list) --> int
    Returns the best dropoff point given a graph and a list of integers representing the homes
    G is a networkx graph
    Subset is list of nodes in a cluster from which we want to find the best drop-off point
    '''
    sum_distances = {}
    for vertex in subset:
        vertex_distances = nx.algorithms.shortest_paths.generic.shortest_path_length(G, source=vertex, target=None, weight='weight')
        for key in vertex_distances:
            sum_distances[key] = sum_distances.get(key, 0) + vertex_distances[key]

    bestDist = float('inf')
    bestNode = 0
    for node in sum_distances:
        if sum_distances[node] < bestDist:
            bestDist = sum_distances[node]
            bestNode = node

    return bestNode

def best_dropoff_efficient(graph, subset, all_pairs_distances):
    """
    return the best dropoff point for the given subset of vertices
    """
    sum_distances = {}
    for vertex in graph.nodes:
        sum_distances[vertex] = 0
    for vertex in subset:
        for target in graph.nodes:
            sum_distances[target] += all_pairs_distances[vertex][target]

    bestDist = np.Inf
    bestNode = 0
    for node in sum_distances.keys():
        if sum_distances[node] < bestDist:
            bestDist = sum_distances[node]
            bestNode = node

    assert not np.isinf(bestDist), "Best dropoff was not found!"

    return bestNode
