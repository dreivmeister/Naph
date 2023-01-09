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
        

class TreeNode(NodeInterface):
    def __init__(self, val, children=None):
        self.val = val
        self.children = children # list of TreeNode objects
    
    def get_val(self):
        return self.val
    
    def get_children(self):
        return self.children
        
    
    
class BinaryTreeNode(NodeInterface):
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left_child = left
        self.right_child = right
    
    def get_val(self):
        return self.val
    
    def get_left(self):
        return self.left_child
    
    def get_right(self):
        return self.right_child
    
    
    
    