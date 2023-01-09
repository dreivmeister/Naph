from node import GraphNode
from ast import literal_eval


def load_graph(path):
    weighted = False
    vals = False
    directed = False
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
    
