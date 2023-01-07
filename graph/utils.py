from node import GraphNode
from ast import literal_eval


def load_graph(path):
    weighted = False
    vals = False
    graph_nodes = []
    with open(path) as file:
        for i,line in enumerate(file):
            # remove comments
            line = line.split('#')[0].rstrip()
            if i == 0:
                weighted = line
                continue
            if i == 1:
                vals = line
                continue
            if line.strip() == '':
                continue
            # handle all 4 cases
            if weighted == 'False' and vals == 'False':
                id, n = line.split(sep=' ')
                if n == '-1':
                    neighbors = []
                else:
                    neighbors = [int(i) for i in n.split(sep=',')]
                graph_nodes.append(GraphNode(id=int(id),neighbors=neighbors))
            if weighted == 'True' and vals == 'False':
                id, n = line.split(sep=' ')
                neighbors = [literal_eval(i) for i in n.split(sep=';')]
                graph_nodes.append(GraphNode(id=int(id),neighbors=neighbors))
                
    # for i in range(len(graph_nodes)):
    #     graph_nodes[i].print_data()
    
    return sorted(graph_nodes, key=lambda x: x.get_id())
   
   
def compute_incoming_edges(graph):
    # assume unweighted graph (list of GraphNode object)
    L = []
    for i in range(len(graph)):
        v = graph[i].get_id()
        inc_edges = []
        for j in range(len(graph)):
            n = graph[j].get_id()
            e = graph[j].get_neighbors()
            if v in e and v != n:
                inc_edges.append(n)
        L.append([v, inc_edges])
    return L
    
