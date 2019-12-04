import numpy as np
import networkx as nx
import tsp_routines
import operator

def map_to_clusters(graph,cluster_centers):
    """
    returns a dict that maps each cluster center to a list of vertices which are closest
    to the given cluster center. Also returns a dict that maps each vertex to the distance 
    to the nearest cluster center
    Inputs:
    graph - fully connected weighted input graph
    cluster_centers - list of vertices which are cluster centers
    """
    # assert tsp_routines.is_fully_connected(graph), "Input graph has to be fully connected"

    mapping = dict()
    for node in cluster_centers:
        mapping[node] = []

    minimum_distance_to_centers = dict()
    for node in graph.nodes:
        distane_to_centers = [graph[node][center]['weight'] for center in cluster_centers]
        minimum_distance = min(distane_to_centers)
        minimum_distance_to_centers[node] = minimum_distance
        nearest_idx = distane_to_centers.index(minimum_distance)
        nearest_center = cluster_centers[nearest_idx]
        mapping[nearest_center].append(node)

    return mapping, minimum_distance_to_centers

def all_k_clusters(graph,num_clusters):
    """
    returns the clustering for each cluster value from 1 to num_clusters
    """
    assert graph.number_of_nodes() > 0, "Input graph is empty"
    assert num_clusters <= graph.number_of_nodes(), "Max number of clusters is the number of vertices in graph"
    num_clusters_to_clustering = dict()
    cluster_centers = []

    startnode = np.random.choice(list(graph.nodes))
    cluster_centers.append(startnode)
    mapping, minimum_distance_to_centers = map_to_clusters(graph,cluster_centers)

    num_clusters_to_clustering[1] = mapping

    for k in range(2,num_clusters+1):
        next_center = max(minimum_distance_to_centers, key = minimum_distance_to_centers.get)
        cluster_centers.append(next_center)
        mapping, minimum_distance_to_centers = map_to_clusters(graph,cluster_centers)
        num_clusters_to_clustering[k] = mapping

    return num_clusters_to_clustering

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

    bestNode = min(sum_distances.items(), key = operator.itemgetter(1))[0]

    # bestDist = np.Inf
    # bestNode = 0
    # for node in sum_distances.keys():
    #     if sum_distances[node] < bestDist:
    #         bestDist = sum_distances[node]
    #         bestNode = node
    # assert not np.isinf(sum_distances[bestNode]), "Best dropoff was not found!"

    return bestNode