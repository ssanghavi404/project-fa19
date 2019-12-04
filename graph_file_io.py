import networkx as nx

def graph_from_input(filename):
    '''graph_from_input(str) --> nx.graph G, int source, np.ndarray[int] homes, dict locToIndex
    Returns a graph created by reading the input file, with integer vertex labels
    Returns list of the home indices
    Returns a map from integer to the name associated with that node'''
    with open(filename, 'r') as f:
        G = nx.Graph()

        locToIndex = {} # maps location name to its index number
        homes = []
        lines = f.readlines()

        numLocations = int(lines[0])
        numTAs = int(lines[1])
        locations = lines[2].split()

        i = 0
        assert len(locations) == numLocations, "Number of locations must match specified value"
        for loc in locations:
            G.add_node(i)
            locToIndex[loc] = i
            i += 1

        TAhomes = lines[3].split()
        assert len(TAhomes) == numTAs, "Number of TA homes must match specified value"
        for home in TAhomes:
            homes.append(locToIndex[home])

        source = locToIndex[lines[4].strip()]

        row = 0
        for line in lines[5:]:
            line = line.split()
            for col in range(len(line)):

                if line[col] != 'x':
                    G.add_edge(row, col)
                    weight = float(line[col])
                    G[row][col]['weight'] = weight
            row += 1

        indexToLoc = {v: k for k, v in locToIndex.items()}
        return G, source, homes, indexToLoc

def write_output_file(path, dropOffs, indexToName, filename):
    '''path is a list of integers that we follow in the car
    dropOffs is a dictionary (int -> [home, home, ...] ) mapping nodes to the homes of the TAs that get off at that node
    indexToName is a list of names corresponding to each index
    filename is a string filename that we write to
    '''
    with open(filename, 'w') as f:
        for step in path:
            f.write(indexToName[step] + " ")

        f.write("\n")
        count = 0
        for key in dropOffs:
            if dropOffs[key] != []:
                count += 1
        f.write(str(count) + "\n")

        for key in dropOffs:
            if dropOffs[key] != []:
                f.write(indexToName[key] + " ")
                for elt in dropOffs[key]:
                    f.write(indexToName[elt] + " ");
                f.write("\n")
