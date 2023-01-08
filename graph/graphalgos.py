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
                    parents[neigh] = v
                    Q.append(neigh)
        return None
    
    visited = set()
    return bfs(start, goal, visited)


def topological_sort_kahn(graph):
    from collections import deque
    n = len(graph)
    in_degree = [0]*n
    for i in range(n):
        for to in graph[i].get_neighbors():
            in_degree[to] += 1
    Q = deque()
    for i in range(n):
        if in_degree[i] == 0:
            Q.append(i)
    
    index = 0
    topological_order = [0]*n
    while len(Q) > 0:
        at = Q.popleft()
        topological_order[index] = at
        index += 1
        for to in graph[at].get_neighbors():
            in_degree[to] -= 1
            if in_degree[to] == 0:
                Q.append(to)
    if index != n:
        return None
    return topological_order
   
   
   
def get_min_dist_node(queue, dist):
    min_dist = float("inf")
    min_node = -1
    for u in queue:
        if dist[u] < min_dist:
            min_dist = dist[u]
            min_node = u
    return min_node
        

def dijkstras_algorithm(graph, source):
    n = len(graph)
    dist = [float("inf")]*n
    prev = [None]*n
    
    Q = []
    for v in graph:
        Q.append(v.get_id())
    dist[source] = 0
    
    while len(Q) > 0:
        u = get_min_dist_node(Q, dist)
        print(u)
        Q.remove(u)
        
        for v in graph[u].get_neighbors():
            id_v, w_uv = v
            if id_v not in Q: # v[0] is node id of v
                continue
            alt = dist[u] + w_uv # v[1] is edge weight between u and v
            if alt < dist[id_v]:
                dist[id_v] = alt
                prev[id_v] = u
    return dist, prev


def add_edges(graph, node, queue, vis):
    vis[node] = True # mark node as visited
    
    edges = graph[node].get_neighbors()
    for edge in edges:
        to, w = edge
        if vis[to] != True:
            from_to_w = (node,) + edge
            queue.append(from_to_w)
    #return queue, vis

def get_min_node(queue):
    min_w = float("inf")
    min_node = None
    for node in queue:
        if node[2] < min_w:
            min_w = node[2]
            min_node = node
    return min_node

def prims_algorithm(graph, start):
    n = len(graph) # num vertices graph
    m = n-1 # num vertices mst
    edgeCount, mstCost = 0,0
    pq = [] # priority queue
    visited = [False]*n # visited array for graph
    mstEdges = [None]*m # stores the edges in the mst
    add_edges(graph, start, pq, visited)
    
    while len(pq) > 0 and edgeCount != m:
        edge = get_min_node(pq)
        pq.remove(edge)
        nodeIndex = edge[1]
        
        if visited[nodeIndex]:
            continue
        
        mstEdges[edgeCount] = edge
        edgeCount += 1
        mstCost += edge[2]
        
        add_edges(graph, nodeIndex, pq, visited)
        
    if edgeCount != m:
        return None
    return mstCost, mstEdges



if __name__=='__main__':
    graph = load_graph('graphconfig.txt')
    #nodes = single_source_shortest_path(graph, graph[2].get_id(), graph[-1].get_id())
    #print(nodes) #[0,2,3,5]
    
    # dist, prev = dijkstras_algorithm(graph, 0)
    # print(dist)
    
    #print(topological_sort_kahn(graph))
    
    c, edges = prims_algorithm(graph, 0)
    print(c)
    print(edges)
    
    
    