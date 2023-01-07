class NodeInterface:
    pass


class GraphNode(NodeInterface):
    def __init__(self, id, neighbors, val=None, weighted=False):
        self.id = id
        # can be None
        self.val = val
        # contains ids of neighbor nodes (unweighted graph)
        # contains (id,weight) tuple of neighbor nodes (weighted graph)
        self.neighbors = neighbors #outgoing edges
        self.incoming_edges = None
        self.weighted = weighted
    
    def get_neighbors(self):
        return self.neighbors
    
    def get_id(self):
        return self.id
    
    def get_val(self):
        # can return None
        return self.val
    
    def print_data(self):
        print(f"id: {self.id} neighs: {self.neighbors}")
        

        
        
        

class Graph:
    def __init__(self, graph_nodes, val=None, weighted=False):
        self.graph_nodes = graph_nodes
        self.val = val
        self.weighted = weighted
        
    