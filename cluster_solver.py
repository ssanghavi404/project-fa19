import tsp_routines
import clustering.funcs
import graph_file_io
import cluster_solver_utils
import numpy as np
import time
import matplotlib.pyplot as plt
import networkx as nx
import cProfile
import os

def add_vertex_to_clusters(clusters,vertex):
    """
    add the given vertex to each cluster.
    Input:
    clusters - dict where the keys are vertices which are cluster centers and the values are a list of
                vertices belonging to this cluster
    vertex - the vertex to be added to each list in `clusters`
    """
    for key in clusters:
        clusters[key].append(vertex)

def get_dropoff_vertices(G, clusters):
    """
    return a list of vertices that is the best dropoff point for each
    of the clusters of homes in clusters.
    Input:
    G - input graph
    clusters - dict whose keys are the cluster centers and values are list of
        vertices belonging to the cluster defined by the key
    """
    best_dropoffs = []
    for key in clusters:
        dropoff = clustering.funcs.best_dropoff(G,clusters[key])
        best_dropoffs.append(dropoff)
    return best_dropoffs

def get_dropoff_vertices_efficient(G, clusters, all_pairs_distances):
    """
    return a list of vertices that is the best dropoff point for each
    of the clusters of homes in clusters.
    Input:
    G - input graph
    clusters - dict whose keys are the cluster centers and values are list of
        vertices belonging to the cluster defined by the key
    """
    best_dropoffs = []
    for key in clusters:
        dropoff = clustering.funcs.best_dropoff_efficient(G,clusters[key],all_pairs_distances)
        best_dropoffs.append(dropoff)
    return best_dropoffs

def solve_by_clustering(graph,homes_subgraph,source,num_clusters):
    """
    return the route to be followed by the car as it drops off TAs.
    Inputs:
    graph - input graph
    homes_subgraph - complete graph with the homes as vertices and shortest
        paths as edge weights
    source - vertex in `graph` that is the start and end of the path followed
        by the car
    num_clusters - the number of clusters to be used to group the homes together
    """
    home_clusters = clustering.funcs.k_cluster(homes_subgraph,num_clusters)
    # The source vertex is added to each of the clusters before determining the best dropoff location.
    # This is done so that vertices that are closer to the source are given higher preference as dropoff points.

    add_vertex_to_clusters(home_clusters,source)
    dropoff_vertices = get_dropoff_vertices(graph, home_clusters)
    # Add the source to the dropoff vertices
    dropoff_vertices.append(source)
    # Get rid of any repeating entries in the dropoff vertices
    dropoff_vertices = list(set(dropoff_vertices))
    # Construct the fully connected sub-graph with the dropoff vertices
    # on which TSP is computed
    dropoff_subgraph = tsp_routines.complete_shortest_path_subgraph(graph,dropoff_vertices)
    tsp_route = tsp_routines.metric_mst_tsp(dropoff_subgraph,source)
    final_path = tsp_routines.tsp_solution_to_path(graph,tsp_route)
    return final_path

def solve_clusters(graph,home_clusters,source,all_pairs_distances):
    add_vertex_to_clusters(home_clusters,source)
    
    dropoff_vertices = get_dropoff_vertices_efficient(graph, home_clusters, all_pairs_distances)
    # dropoff_vertices = get_dropoff_vertices(graph,home_clusters)

    # Add the source to the dropoff vertices
    dropoff_vertices.append(source)
    # Get rid of any repeating entries in the dropoff vertices
    dropoff_vertices = list(set(dropoff_vertices))
    # Construct the fully connected sub-graph with the dropoff vertices
    # on which TSP is computed
    dropoff_subgraph = tsp_routines.complete_shortest_path_subgraph_efficient(graph,dropoff_vertices,all_pairs_distances)
    tsp_route = tsp_routines.metric_mst_tsp(dropoff_subgraph,source)
    final_path = tsp_routines.tsp_solution_to_path(graph,tsp_route)
    return final_path

def solver(graph,homes,source,homes_subgraph,num_clusters):
    """
    returns the cost, dropoffs, route for given number of clusters.
    Inputs:
    graph - the input graph
    homes - list of homes in graph
    source - source vertex in graph
    homes_subgraph - complete graph with vertices as homes and shortest path
        between homes in `graph` as the edge weights
    num_clusters - number of clusters to be used in homes
    """
    route = solve_by_clustering(graph,homes_subgraph,source,num_clusters)
    dropoffs = cluster_solver_utils.nearest_dropoff(graph,route,homes)
    cost = cluster_solver_utils.eval_cost(graph,route,dropoffs)
    return cost,dropoffs,route

def solver_efficient(graph,homes,source,home_clusters,all_pairs_distances):
    """
    returns the cost, dropoffs, route for the given clustering
    Inputs:
    graph - input graph
    homes - list of homes
    source - source vertex
    home_clusters - dict such that home_clusters[v1] gives all the homes belonging
        to v1's cluster where v1 is a cluster center of homes
    all_pairs_distances - shortest path distance between any pair of vertices
    """
    route = solve_clusters(graph,home_clusters,source,all_pairs_distances)
    dropoffs = cluster_solver_utils.nearest_dropoff_efficient(graph,route,homes,all_pairs_distances)
    cost = cluster_solver_utils.eval_cost(graph,route,dropoffs)
    return cost, dropoffs, route

def optimal_num_clusters(graph,homes,source,homes_subgraph):
    num_homes = len(homes)
    cost_list = []
    cluster_list = range(1,num_homes+1)
    for num_clusters in cluster_list:
        cost,dropoffs,route = solver(graph,homes,source,homes_subgraph,num_clusters)
        cost_list.append(cost)
    idx = cost_list.index(min(cost_list))
    return cluster_list[idx]

def optimal_route(graph,homes,source):
    number_of_homes = len(homes)
    all_pairs_distances = dict(nx.shortest_path_length(graph, weight = 'weight'))
    homes_subgraph = tsp_routines.complete_shortest_path_subgraph_efficient(graph,homes,all_pairs_distances)
    all_home_clusters = clustering.funcs.all_k_clusters(homes_subgraph,number_of_homes,all_pairs_distances)
    
    cluster_list = range(1,number_of_homes+1)
    optimal_cost = np.Inf
    optimal_dropoffs = dict()
    optimal_route = []
    optimal_num_clusters = 0


    for num_clusters in cluster_list:
        home_clusters = all_home_clusters[num_clusters]
        cost, dropoffs, route = solver_efficient(graph,homes,source,home_clusters,all_pairs_distances)
        if cost < optimal_cost:
            optimal_cost = cost
            optimal_route = route 
            optimal_dropoffs = dropoffs
            optimal_num_clusters = num_clusters

    return optimal_cost, optimal_dropoffs, optimal_route, optimal_num_clusters

def cost_vs_clusters(filename):
    """
    compute the cost for number of clusters = 1 to number of homes
    return cost and number of clusters
    """
    graph,source,homes,indexToLoc = graph_file_io.graph_from_input(filename)
    num_homes = len(homes)
    homes_subgraph = tsp_routines.complete_shortest_path_subgraph(graph,homes)

    cost_list = []
    num_clusters_list = range(2,num_homes)
    for num_clusters in num_clusters_list:
        cost,dropoffs,route = solver(graph,homes,source,homes_subgraph,num_clusters)
        cost_list.append(cost)
    return num_clusters_list, cost_list

def plot_cost_vs_clusters(cost,num_clusters,filename):
    fig, ax = plt.subplots()
    ax.plot(num_clusters,cost)
    ax.grid()
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Cost of dropping off TAs")
    fig.savefig(filename)
    plt.close()

# directory_name = "inputs/"
# directory = os.fsencode(directory_name)
# for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     if filename.endswith("50.in"):
#         print("Solving : ", filename)
#         inputfile = directory_name + filename
#         num_clusters, cost = cost_vs_clusters(inputfile)
#         outfile = "plots/" + filename.strip(".in") + ".png"
        # plot_cost_vs_clusters(cost,num_clusters,outfile)

filename = "inputs/194_100.in"
graph,source,homes,indexToLoc = graph_file_io.graph_from_input(filename)
number_of_homes = len(homes)
print("Number of homes = ", number_of_homes)

all_pairs_distances = dict(nx.shortest_path_length(graph, weight = 'weight'))
homes_subgraph = tsp_routines.complete_shortest_path_subgraph_efficient(graph,homes,all_pairs_distances)

all_home_clusters = clustering.funcs.all_k_clusters(homes_subgraph,number_of_homes,all_pairs_distances)

cost, dropoffs, route, num_clusters = optimal_route(graph,homes,source)