import networkx as nx

def eval_cost(G, path, dropoffs):
    '''
    eval_cost(nx.graph, np.ndarray, dict) -> int
    path is a iterable of integers that we follow in the car
    dropOffs is a dictionary (int -> [home, home, ...] )
    '''
    assert isinstance(G, nx.Graph) , "G must be a graph"
    assert hasattr(path, '__iter__') , "path must be an array of integers"
    assert isinstance(dropoffs, dict) , "dropoffs must be a dictionary mapping node to homes"

    cost = 0
    prevNode = path[0]
    for node in path[1:]:
        cost += 2/3 * G[prevNode][node]['weight'] # weight of the edge from previous to next node
        prevNode = node

        for home in dropoffs[node]:
            cost += nx.astar_path_length(G, node, home)

    return cost

def eval_cost_efficient(G, path, dropoffs, all_pairs_distances):
    '''
    eval_cost(nx.graph, np.ndarray, dict) -> int
    path is a iterable of integers that we follow in the car
    dropOffs is a dictionary (int -> [home, home, ...] )
    all_pairs_distances - shortest path distance between any pair of vertices
    '''

    cost = 0
    prevNode = path[0]
    for node in path[1:]:
        cost += 2/3 * G[prevNode][node]['weight'] # weight of the edge from previous to next node
        prevNode = node

    for node in path:
        for home in dropoffs[node]:
            cost += all_pairs_distances[node][home]

    return cost


def nearest_dropoff(G,route,homes):
    """
    Returns a list representing the nearest dropoff point for each home
    G - graph with weights specified as distances
    route - array of indices representing vertices on path of Car
    homes - array of indices representing vertices which are TA homes
    """
    shortest_path_lens = dict(nx.all_pairs_dijkstra_path_length(G))

    wheredict = dict()
    for node in route:
        wheredict.update({node:[]}) #initialize a dropoff at each node in route. there could be places where noone is dropped.

    for h in homes:
        ls = shortest_path_lens[h]#shortest path from h to all nodes (dict)
        ls = {k:ls[k] for k in route}
        wheredict[min(ls, key=ls.get)].append(h)

    return wheredict

def nearest_dropoff_efficient(graph,route,homes,all_pairs_distances):
    """
    Returns a list representing the nearest dropoff point for each home
    graph - graph with weights specified as distances
    route - array of indices representing vertices on path of Car
    homes - array of indices representing vertices which are TA homes
    all_pairs_distances - dict such that all_pairs_distances[v1][v2] gives length of shortest path from v1 to v2
    """

    wheredict = dict()
    for node in route:
        wheredict.update({node:[]}) #initialize a dropoff at each node in route. there could be places where noone is dropped.

    for h in homes:
        dist_to_route = [all_pairs_distances[h][r] for r in route]
        min_dist = min(dist_to_route)
        idx = dist_to_route.index(min_dist)
        best_dropoff = route[idx]
        wheredict[best_dropoff].append(h)

    return wheredict