import networkx as nx


def is_fully_connected(G):
    """
    returns true if the graph G is (a) undirected (b) is fully connected.
    """
    N = G.number_of_nodes()
    E = G.number_of_edges()
    if isinstance(G,nx.Graph) and 2*E == N*(N-1):
        return True
    else:
        return False

def drop_repeated(input_list):
    """
    return a list omitting the repeated elements of input_list while maintaining the order of
    elements in the list.
    """
    output_list = []
    for entry in input_list:
        if entry in output_list:
            pass
        else:
            output_list.append(entry)
    return output_list

def tsp_solution_to_path(G,tsp_route):
    """
    converts the given tsp sequence to be followed by the car into a
    path in the graph G
    Input:
    G - undirected weighted input graph
    tsp_route - list of vertices specifying the route to be followed by the car
    """
    prev = tsp_route[0]
    final_path = []
    final_path.append(prev)
    for vertex in tsp_route[1:]:
        path = nx.shortest_path(G,prev,vertex,weight='weight')
        final_path += path[1:]
        prev = vertex
    return final_path

#################################################################################
# TSP approximation algorithm via minimum spanning trees
def metric_mst_tsp(G,s):
    """
    Approximate TSP algorithm. Approximation factor = 2.0
    Returns a list of vertices of G representing an approximate Traveling Salesman Problem cycle
    starting and ending at the source s. Uses an MST approximation.
    Inputs:
    G -- a fully connected undirected weighted graph where edge weights satisfy triangle inequality.
    s -- a vertex in G
    """
    T = nx.minimum_spanning_tree(G)
    dfs_edges = list(nx.dfs_edges(T,source=s))
    vertices = []
    for e in dfs_edges:
        vertices.append(e[0])
        vertices.append(e[1])
    tsp_path = drop_repeated(vertices)
    tsp_path.append(s)

    return tsp_path
#################################################################################


#################################################################################
# Routines for Christofides TSP approximation algorithm
def find_max_weight(G):
    """
    return the weight of the heaviest edge in G
    """
    max_weight = 0
    for edge in G.edges:
        u,v = edge[0],edge[1]
        if G[u][v]['weight'] > max_weight:
            max_weight = G[u][v]['weight']
    return max_weight

def transform_graph_for_max_matching(G):
    """
    returns a graph with modified edge weights such that a maximal matching on the
    modified graph corresponds to a minimal matching on G
    """
    max_weight = find_max_weight(G)
    modified_graph = nx.Graph()
    for edge in G.edges:
        u,v = edge[0],edge[1]
        wt = G[u][v]['weight']
        modified_graph.add_edge(u,v,weight=max_weight-wt)
    return modified_graph

def min_weight_matching(G):
    """
    Returns a set of edges representing a minimum weight matching.
    Every node appears only once in a matching.
    """
    modified_graph = transform_graph_for_max_matching(G)
    min_matching = nx.max_weight_matching(modified_graph)
    return min_matching

def find_odd_degree_nodes(G):
    """
    returns a list of vertices which have odd degree in graph G.
    """
    degree = G.degree()
    odd_nodes = []
    for v in G.nodes:
        if degree[v] % 2 == 0:
            pass
        else:
            odd_nodes.append(v)
    return odd_nodes

def construct_fully_connected_subgraph(node_subset,G):
    """
    return a graph with `node_subset` as the nodes.
    The graph is fully connected and uses the same edge weight as in G.
    Inputs:
    G - a fully connected graph with edge weights
    node_subset - List with a subset of nodes to be used in the new graph
    """
    sub_graph = nx.Graph()
    for u in node_subset:
        for v in node_subset:
            if u == v:
                pass
            else:
                wt = G[u][v]['weight']
                sub_graph.add_edge(u,v,weight=wt)
    return sub_graph

def complete_shortest_path_subgraph(G, subset):
    """
    return a fully connected graph using the vertices in `subset`
    and whose edges are weighted by the shortest path between these
    vertices in the graph `G`
    """
    new_graph = nx.Graph()
    distances = dict(nx.all_pairs_dijkstra_path_length(G))
    for I in range(len(subset)-1):
        u = subset[I]
        for J in range(I+1,len(subset)):
            v = subset[J]
            dist = distances[u][v]
            new_graph.add_edge(u,v,weight=dist)
    return new_graph

def complete_shortest_path_subgraph_efficient(graph,subset,distances):
    """
    return a fully connected graph using the vertices in `subset`
    and whose edges are weighted by the shortest path between these
    vertices in the graph `G`
    Inputs:
    graph - input graph
    subset - list of vertices to be used for the subgraph
    distances - dictionary such that distances[v1][v2] is the length of shortest path from
        v1 to v2
    """
    new_graph = nx.Graph()
    for I in range(len(subset)):
        u = subset[I]
        for J in range(I,len(subset)):
            v = subset[J]
            dist = distances[u][v]
            new_graph.add_edge(u,v,weight=dist)
    return new_graph

def add_specified_edges(G1,G2,edges):
    """
    copy G2[edges] into G1
    """
    for edge in edges:
        u,v = edge[0],edge[1]
        wt = G2[u][v]['weight']
        G1.add_edge(u,v,weight=wt)

def construct_eulerian_multigraph(G1,G2,edges):
    """
    Construct a multigraph M:
        (1) copy G1 into M
        (2) copy specified edges of G2 into M
    """
    eulerian_multigraph = nx.MultiGraph()
    add_specified_edges(eulerian_multigraph,G1,G1.edges)
    add_specified_edges(eulerian_multigraph,G2,edges)
    return eulerian_multigraph

def metric_christofides_tsp(G,s):
    """
    return a list of vertices of G representing an approximate Traveling Salesman Problem cycle
    starting and ending at s. Uses the Christofides approximation.
    G -- a fully connected undirected weighted graph where edge weights satisfy triangle inequality.
    s -- a vertex in G
    """
    tree = nx.minimum_spanning_tree(G)
    odd_nodes = find_odd_degree_nodes(tree)
    sub_graph = construct_fully_connected_subgraph(odd_nodes,G)
    min_matching_edges = min_weight_matching(sub_graph)
    eulerian_graph = construct_eulerian_multigraph(tree,G,min_matching_edges)
    circuit = nx.eulerian_circuit(eulerian_graph,s)
    vertices = []
    for edge in circuit:
        vertices.append(edge[0])
        vertices.append(edge[1])
    christofides_tsp = drop_repeated(vertices)
    christofides_tsp.append(s)
    return christofides_tsp
