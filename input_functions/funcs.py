import numpy as np
import pandas as pd
import networkx as nx

import itertools


__all__ = ('check_triangle','create_new_graph','path_in_newG_to_old')

def path_in_newG_to_old(newpath, edge_to_path_mapper):
    """takes a path in the newG and returns the old"""

    edges = []
    for k, newnode in enumerate(newpath):
        if k == 0:
            continue
        edges.append((newpath[k-1], newnode))
    oldpath = []
    for i, edge in enumerate(edges):
        if i == 0: #in this case we don't want to eliminate the first node. after this we need to other wise we have duplicate nodes in seq
            oldpath.append(edge_to_path_mapper[edge])
        else:
            oldpath.append(edge_to_path_mapper[edge][1:])
    oldpath = np.concatenate(oldpath )

    return oldpath

def create_new_graph(G, dropoffnodes):
    """
    dropoffnodes here includes SODA!

    returns
    newG - fully connected graph with edge weights given by the shortest path between drop off points
    nodemapper - dict, maps nodes in newG to nodes in G
    edge_to_path_mapper - dict, maps edges in newG to paths in G
    """
    ndropoffs = len(dropoffnodes)

    edge_to_path_mapper = dict() #maps new edges to old paths
    nodemapper = dict() #maps new nodes to old nodes
    reversenodemapper = dict() #maps old nodes to new nodes

    for i, node in enumerate(dropoffnodes):
        nodemapper.update({i:node})
        reversenodemapper.update({node:i})

    newmat = np.zeros((ndropoffs, ndropoffs))
    for nodes in itertools.product(dropoffnodes, dropoffnodes):
        node1, node2 = nodes
        newnode1, newnode2 = reversenodemapper[node1], reversenodemapper[node2]
        spath = nx.algorithms.shortest_paths.weighted.dijkstra_path(G,node1, node2)
        pathlen = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(G,node1, node2)
        newmat[newnode1, newnode2] = pathlen
        edge_to_path_mapper.update({(newnode1, newnode2):spath})



    newG = nx.from_numpy_array(newmat)

    return newG, nodemapper, edge_to_path_mapper

def check_triangle(G, **kwargs):
    adjarr = nx.to_numpy_array(G)
    for edge in list(G.edges):
        shortestd = nx.algorithms.shortest_paths.weighted.dijkstra_path_length(G, edge[0], edge[1])
        edged = adjarr[edge[0], edge[1]]
        if shortestd < edged:
            return False, edge
    return True

def connect_graph(G,**kwargs):
    """connects a graph. All edges will be length 1 between connected component
    graph and point locaitons (for triangle inequality) as input

    returns a graph
    """
    adjmat = nx.to_numpy_matrix(G)
    concomp = nx.algorithms.components.connected_components(G)
    choices = []
    for comp in concomp:
        choices.append(np.random.choice(list(comp), 1, replace = False)[0])
    choices = np.array(choices)
    import itertools
    indecies = np.array(list(itertools.product(choices, choices)))
    setter = []
    for index in indecies:
        if index[0] == index[1]:
            setter.append(False)
        else:
            setter.append(True)
    indecies = indecies[setter]
    for index in indecies:
        adjmat[index[0], index[1]] = 1
        adjmat[index[1], index[0]] = 1

    return nx.from_numpy_matrix(adjmat)

def connect_2clusters(graphs, vs, maxconnections = 1, verbosity = 0, **kwargs):
    """
    array like of of graphs
    array like of points in each graph
    maxconnections is how many connections between clusters
    """
    G1, G2 = graphs
    v1, v2 = vs
    a, b = nx.to_numpy_array(G1), nx.to_numpy_array(G2)
    anew = np.concatenate((a, np.zeros((a.shape[0], b.shape[0]), )),axis = 1)
    bnew = np.concatenate((np.zeros((b.shape[0], a.shape[0]),), b),axis = 1)
    out = np.concatenate((anew,bnew),axis = 0)
    concomplist = list(nx.algorithms.components.connected_components(nx.from_numpy_array(out)))
    if len(concomplist)!= 2:
        raise ValueError('trying to merge more than two graphs')
    ncon = np.random.randint(low = 1, high = maxconnections+1)
    choices_from_first = np.random.choice(list(concomplist[0]), ncon,)
    choices_from_second = np.random.choice(list(concomplist[1]), ncon)
    for inds in zip(choices_from_first, choices_from_second):
        dist = np.linalg.norm(v1[inds[0]] - v2[inds[1]- len(concomplist[0])]) #to get the indexing correct for the points
        out[inds[0], inds[1]] = dist
        out[inds[1], inds[0]] = dist
    G = nx.from_numpy_array(out)
    if not check_triangle(G, **kwargs):
        raise ValueError('not satisfy triangle')
    if not nx.algorithms.components.connected.is_connected(G):
        raise ValueError('not connected')
    v = np.concatenate((v1, v2))
    if verbosity>1:
        print('generated ', ncon, ' connections between clusters')
    return G, v

def generate_cluster(nnodes, p, yrange=(0,1), xrange = (0,1), **kwargs):
    """returns a connected graph and location of points in space for generating a clustered graph

    nnodes - how many nodes in cluster
    p - probability of generating an edge between nodes
    """
    n = nnodes
    G = nx.fast_gnp_random_graph(n, p,)
    adjmat = nx.to_numpy_matrix(G)

    #generate random n random points:
    v = np.random.rand(n, 2)

    v[:,0] = (v[:,0] )*(xrange[1] - xrange[0]) + xrange[0]
    v[:,1] = (v[:,1] )*(yrange[1] - yrange[0]) + yrange[0]

    #check if connected
    G = connect_graph(G, **kwargs)
    adjmat = nx.to_numpy_matrix(G)

    tadjmat = adjmat.copy()
    for i, row in enumerate(adjmat):
        newrow = np.array([np.linalg.norm(v[i] - v[j]) for j in range(n)])*np.array(row)
        tadjmat[i] = newrow

    adjmat = tadjmat
    del tadjmat
    adjmat
    Gp = nx.from_numpy_matrix(adjmat)
    return Gp, v

def generate_clustered_graph(k, nnodes, centers, xspan, yspan,p = .2,**kwargs):
    """
    k - how many clusters
    nnodes - int . how many nodes total
    """
    nper = []
    for i, worker in enumerate(np.sort(np.random.choice(np.linspace(1,nnodes,nnodes), k-1, replace = False))):
        if i == 0:
            last = 0
        nper.append(int(worker - last))
        last = worker
    nper.append(int(nnodes - last))
    xrange = (centers[0][0]+xspan[0]/2, centers[0][0]-xspan[0]/2)
    yrange = (centers[0][1]+yspan[0]/2, centers[0][1]-yspan[0]/2)
    G, v = generate_cluster(nper[0], p, xrange , yrange )
    dropoff = []
    dropoff.append(np.random.choice(list(G.nodes)))
    indexcounter = 0
    for i in range(k):
        if i == 0:
            continue
        indexcounter+=nper[i-1]
        #generate all clusters and merge
        xrange = (centers[i][0]+xspan[0]/2, centers[i][0]-xspan[0]/2)
        yrange = (centers[i][1]+yspan[0]/2, centers[i][1]-yspan[0]/2)
        G1, v1 = generate_cluster(nper[i], .1, xrange, yrange)
        dropoff.append(np.random.choice(list(G1.nodes))+indexcounter)
        G, v = connect_2clusters((G, G1), (v, v1),**kwargs)

    return G, v, dropoff
