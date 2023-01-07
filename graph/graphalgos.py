from node import GraphNode
from utils import load_graph

def nodes_of_graph_dfs(graph, curr):
    def dfs(curr,nodes_of_graph,visited):
        # recursive
        nodes_of_graph.append(curr)
        for neigh in graph[curr].get_neighbors():
            if neigh not in visited:
                visited.add(neigh)
                dfs(neigh, nodes_of_graph, visited)
        return nodes_of_graph
    
    visited = set()
    visited.add(curr)
    nodes = []
    return dfs(curr, nodes, visited)


def nodes_of_graph_bfs(graph, start):
    from collections import deque
    
    def bfs(curr,nodes_of_graph,visited):
        # iterative
        Q = deque()
        visited.add(curr)
        nodes_of_graph.append(curr)
        Q.append(curr)
        while len(Q) > 0:
            v = Q.popleft()
            for neigh in graph[v].get_neighbors():
                if neigh not in visited:
                    visited.add(neigh)
                    nodes_of_graph.append(neigh)
                    Q.append(neigh)
        return nodes_of_graph
    visited = set()
    nodes = []
    return bfs(start, nodes, visited)






if __name__=='__main__':
    graph = load_graph('graphconfig.txt')
    nodes = nodes_of_graph_bfs(graph, graph[0].get_id())
    print(nodes)
    