import tsp_routines
import clustering_routines
import graph_file_io
import cluster_solver_utils
import numpy as np
import time
import matplotlib.pyplot as plt
import networkx as nx
import os
import pickle

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
        dropoff = clustering_routines.best_dropoff_efficient(G,clusters[key],all_pairs_distances)
        best_dropoffs.append(dropoff)
    return best_dropoffs

def get_car_path(graph,home_clusters,source,all_pairs_distances,all_pairs_shortest_paths):
    """
    return the path to be followed by the car that visits the best dropoffs for each cluster of homes
    Inputs:
    graph - input graph
    home_clusters - dict such that home_clusters[v1] gives all the homes belonging
        to v1's cluster where v1 is a cluster center of homes
    source - the source vertex for the car path
    all_pairs_distances - shortest path distance between any pair of vertices
    all_pairs_shortest_paths - shortest path between any pair of vertices 
    """
    add_vertex_to_clusters(home_clusters,source)
    dropoff_vertices = get_dropoff_vertices_efficient(graph, home_clusters, all_pairs_distances)

    # Add the source to the dropoff vertices
    dropoff_vertices.append(source)
    # Get rid of any repeating entries in the dropoff vertices
    dropoff_vertices = list(set(dropoff_vertices))
    # Construct the fully connected sub-graph with the dropoff vertices
    # on which TSP is computed
    dropoff_subgraph = tsp_routines.complete_shortest_path_subgraph_efficient(graph,dropoff_vertices,all_pairs_distances)
    tsp_route = tsp_routines.metric_mst_tsp(dropoff_subgraph,source)
    final_path = tsp_routines.tsp_solution_to_path(graph,tsp_route,all_pairs_shortest_paths)
    return final_path

def solver(graph,homes,source,home_clusters,all_pairs_distances,all_pairs_shortest_paths):
    """
    returns the cost, dropoffs, route for the given clustering
    Inputs:
    graph - input graph
    homes - list of homes
    source - source vertex
    home_clusters - dict such that home_clusters[v1] gives all the homes belonging
        to v1's cluster where v1 is a cluster center of homes
    all_pairs_distances - shortest path distance between any pair of vertices
    all_pairs_shortest_paths - shortest path between any pair of vertices
    """
    car_path = get_car_path(graph,home_clusters,source,all_pairs_distances,all_pairs_shortest_paths)
    dropoffs = cluster_solver_utils.nearest_dropoff_efficient(graph,car_path,homes,all_pairs_distances)
    cost = cluster_solver_utils.eval_cost_efficient(graph,car_path,dropoffs,all_pairs_distances)
    return cost, dropoffs, car_path


def optimal_route(graph,homes,source):
    """
    returns - optimal_cost, optimal_dropoffs, optimal_route, optimal_number_of_clusters

    Iterate over all number of clusters (from 1 to len(homes)). Solve the problem for each number of clusters. Pick the best solution.
    Inputs:
    graph - input graph
    homes - list of homes
    source - source vertex for the car path
    """
    number_of_homes = len(homes)
    all_pairs_distances = dict(nx.shortest_path_length(graph, weight = 'weight'))
    all_pairs_shortest_paths = dict(nx.shortest_path(graph, weight = 'weight'))
    homes_subgraph = tsp_routines.complete_shortest_path_subgraph_efficient(graph,homes,all_pairs_distances)
    num_clusters_to_clustering = clustering_routines.all_k_clusters(homes_subgraph,number_of_homes)
    
    cluster_list = range(1,number_of_homes+1)
    optimal_cost = np.Inf
    optimal_dropoffs = dict()
    optimal_route = []
    optimal_num_clusters = 0


    for num_clusters in cluster_list:
        home_clusters = num_clusters_to_clustering[num_clusters]
        cost, dropoffs, route = solver(graph,homes,source,home_clusters,all_pairs_distances,all_pairs_shortest_paths)
        if cost < optimal_cost:
            optimal_cost = cost
            optimal_route = route 
            optimal_dropoffs = dropoffs
            optimal_num_clusters = num_clusters

    return optimal_cost, optimal_dropoffs, optimal_route, optimal_num_clusters

def get_named_dict(numbered_dict,num_to_name):
    """
    returns a dict equivalent to the numbered_dict but replaces all keys and values
    by the name associated with the number in num_to_name.
    Use this to map a numeric solution from the algorithm to a named solution for output.
    Inputs:
    numbered_dict - a dict whose keys and values are valid keys in num_to_name
    num_to_name - a dict mapping numbers to names
    """
    named_dict = dict()
    for num_key in numbered_dict.keys():
        name_key = num_to_name[num_key]
        num_values = numbered_dict[num_key]
        named_dict[name_key] = [num_to_name[v] for v in num_values]
    return named_dict

def solve_inputfile(inputfile):
    """
    return the optimal route and dropoffs for the given input file. The route and dropoffs use the location names
    specified in the input file.
    """
    graph,source,homes,indexToLoc = graph_file_io.graph_from_input(inputfile)
    cost, dropoffs, route, num_clusters = optimal_route(graph,homes,source)
    named_route = [indexToLoc[r] for r in route]
    named_dropoffs = get_named_dict(dropoffs,indexToLoc)
    return named_route, named_dropoffs

def cost_vs_clusters(filename):
    """
    compute the cost for number of clusters = 1 to number of homes
    return cost and number of clusters
    """
    graph,source,homes,indexToLoc = graph_file_io.graph_from_input(filename)

    number_of_homes = len(homes)
    all_pairs_distances = dict(nx.shortest_path_length(graph, weight = 'weight'))
    all_pairs_shortest_paths = dict(nx.shortest_path(graph, weight = 'weight'))
    homes_subgraph = tsp_routines.complete_shortest_path_subgraph_efficient(graph,homes,all_pairs_distances)
    num_clusters_to_clustering = clustering_routines.all_k_clusters(homes_subgraph,number_of_homes)
    
    cluster_list = range(1,number_of_homes+1)
    cost_list = []

    for num_clusters in cluster_list:
        home_clusters = num_clusters_to_clustering[num_clusters]
        cost, dropoffs, route = solver(graph,homes,source,home_clusters,all_pairs_distances,all_pairs_shortest_paths)
        cost_list.append(cost)
    return cluster_list, cost_list

def plot_cost_vs_clusters(cost,num_clusters,filename):
    """
    plots the cost against the number of clusters and saves the plot to filename
    """
    fig, ax = plt.subplots()
    ax.plot(num_clusters,cost)
    ax.grid()
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Cost of dropping off TAs")
    fig.savefig(filename)
    plt.close()

def generate_all_cost_plots(suffix):
    """
    Given a suffix (i.e. "200.in", "100.in", "50.in") iterate through all the inputfiles
    of this type. Compute the cost for all possible clusterings. Generate the plots and 
    save them to "plots/50 or 100 or 200/filename.png
    """
    directory_name = "inputs/"
    directory = os.fsencode(directory_name)
    outfolder = "plots/" + suffix.strip(".in") + "/"
    try:
        os.makedirs(outfolder)
    except FileExistsError:
        pass
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(suffix):
            print("Solving : ", filename)
            inputfile = directory_name + filename
            num_clusters, cost = cost_vs_clusters(inputfile)
            outfile = outfolder + filename.strip(".in") + ".png"
            plot_cost_vs_clusters(cost,num_clusters,outfile)

def generate_all_optimal_solutions(suffix):
    """
    Given a suffix (i.e. "200.in", "100.in", "50.in") iterate through all the inputfiles
    of this type. Compute the optimal solution for this file. 
    
    Save the output in 
    "outputs/50 or 100 or 200/filename.pickle

    The pickle when loaded yieds a dictionary such that dict["route"] gives the car route and
    dict["dropoffs"] gives the dropoff locations.
    """
    directory_name = "inputs/"
    directory = os.fsencode(directory_name)
    outfolder = "outputs/" + suffix.strip(".in") + "/"
    try:
        os.makedirs(outfolder)
    except FileExistsError:
        pass
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(suffix):
            print("Solving : ", filename)
            inputfile = directory_name + filename
            route, dropoffs = solve_inputfile(inputfile)
            outputdict = {"route" : route, "dropoffs" : dropoffs}
            outputfile = outfolder + filename.strip(".in") + ".pickle"
            with open(outputfile, "wb") as handle:
                pickle.dump(outputdict, handle, protocol = pickle.HIGHEST_PROTOCOL)

################################################################
##   EXAMPLES
##
##   
# inputfile = "inputs/170_100.in"
# route, dropoffs = solve_inputfile(inputfile)
# generate_all_cost_plots("50.in")
# generate_all_optimal_solutions("100.in")

