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


def single_source_shortest_path(graph, start, goal):
    from collections import deque
    
    def bfs(start, goal, visited):
        Q = deque()
        parents = [-1]*len(graph)
        visited.add(start)
        Q.append(start)
        while len(Q) > 0:
            v = Q.popleft()
            if v == goal:
                shortest_path = [v]
                while parents[v] != -1:
                    shortest_path.append(parents[v])
                    v = parents[v]
                return shortest_path[::-1]
            for neigh in graph[v].get_neighbors():
                if neigh not in visited:
                    visited.add(neigh)
                    # add parent
                    parents[neigh] = v
                    Q.append(neigh)
        return None
    
    visited = set()
    return bfs(start, goal, visited)
        





if __name__=='__main__':
    graph = load_graph('graphconfig.txt')
    nodes = single_source_shortest_path(graph, graph[0].get_id(), graph[-1].get_id())
    print(nodes) #[0,2,3,5]
    