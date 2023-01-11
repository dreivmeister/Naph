def load_grid(path):
    #load grid into list of lists
    grid = []
    with open(path) as file:
        for line in file:
            grid.append(list(line.rstrip()))
    return grid

def print_grid(grid):
    for i in range(len(grid)):
        print(''.join(grid[i]))
        
def show_path(grid, parents, sr, sc):
    while parents[sr][sc] != 'S' and parents[sr][sc] is not None:
        grid[sr][sc] = 'X'
        sr,sc = parents[sr][sc]
    return grid

def shortest_path_grid_bfs(grid, sr=0, sc=1, end_sym='E', block_sym='#'):
    R = len(grid)
    C = len(grid[0])
    rq = []
    cq = []
    
    move_count = 0
    nodes_left = 1
    nodes_next = 0
    
    reached_end = False
    end_c = None
    
    dr = [-1,+1,0,0]
    dc = [0,0,+1,-1]
    
    visited = [[False for j in range(C)] for i in range(R)]
    parent = [[None for j in range(C)] for i in range(R)]
    
    
    #solve 
    rq.append(sr)
    cq.append(sc)
    visited[sr][sc] = True
    while len(rq) > 0:
        r = rq.pop(0)
        c = cq.pop(0)
        if grid[r][c] == end_sym:
            end_c = (r,c)
            reached_end = True
            break
        # explore neighbours
        for i in range(4):
            rr = r+dr[i]
            cc = c+dc[i]
            
            if rr < 0 or cc < 0 or rr >= R or cc >= C:
                continue
            if visited[rr][cc] or grid[rr][cc] == block_sym:
                continue
            
            rq.append(rr)
            cq.append(cc)
            visited[rr][cc] = True
            parent[rr][cc] = (r,c)
            nodes_next += 1
            
        nodes_left -= 1
        if nodes_left == 0:
            nodes_left = nodes_next
            nodes_next = 0
            move_count += 1
    if reached_end:
        return move_count,parent,end_c
    return -1,-1,-1

def get_min_node(nodes,scores):
    min_score = float("inf")
    min_node = None
    for n in nodes:
        if scores[n[0]][n[1]] < min_score:
            min_score = scores[n[0]][n[1]]
            min_node = n
    return min_node

def shortest_path_grid_astar(maze, h, start=(0,1), end=(9,11), block_sym='#'):
    R = len(maze)
    C = len(maze[0])
    dr = [-1,+1,0,0]
    dc = [0,0,+1,-1]
    # priority queue 
    open_set = [start]
    # parents
    came_from = [[None for j in range(len(maze[0]))] for i in range(len(maze))]
    g_score = [[float("inf") for j in range(len(maze[0]))] for i in range(len(maze))]
    g_score[start[0]][start[1]] = 0
    f_score = [[float("inf") for j in range(len(maze[0]))] for i in range(len(maze))]
    f_score[start[0]][start[1]] = g_score[start[0]][start[1]] + h(start,end) # 0 + (start)
    
    while len(open_set) > 0:
        current = get_min_node(open_set,f_score)
        if current == end:
            return show_path(maze,came_from,current[0],current[1]),f_score[current[0]][current[1]]
        
        open_set.remove(current)
        # for each neighbour
        for i in range(4):
            neigh = (current[0]+dr[i],current[1]+dc[i])
            if neigh[0]<0 or neigh[0]>=R or neigh[1]<0 or neigh[1]>=C or maze[neigh[0]][neigh[1]]==block_sym:
                continue
            tent_g_score = g_score[current[0]][current[1]] + 1 # unit distance between nodes
            if tent_g_score < g_score[neigh[0]][neigh[1]]:
                came_from[neigh[0]][neigh[1]] = current
                g_score[neigh[0]][neigh[1]] = tent_g_score
                f_score[neigh[0]][neigh[1]] = tent_g_score + h(neigh, end)
                if neigh not in open_set:
                    open_set.append(neigh)
    return -1,-1
    

def h(node, goal):
    return abs(node[0]-goal[0])+abs(node[1]-goal[1])

if __name__=='__main__':
    
    grid = load_grid('maze.txt')
    # mc,p,ec = shortest_path_grid_bfs(grid)
    # if mc == -1:
    #     raise ValueError('No Solution found!')
    # grid = show_path(grid,p,ec[0],ec[1])
    # print_grid(grid)
    # print(mc)
    
    
    result,cost = shortest_path_grid_astar(grid,h)
    if result == -1:
        raise ValueError('No Solution found!')
    print(cost)
    print_grid(result)