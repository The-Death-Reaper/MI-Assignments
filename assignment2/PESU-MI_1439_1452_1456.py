import heapq


def A_star_Traversal(cost, heuristic, start_point, goals):
    l = []
    heap = []
    heapq.heappush(heap, (0 + heuristic[start_point], start_point, [start_point], 0))
    visited = []
    while(heap):
        node = heapq.heappop(heap)
        if node[1] in goals:
            l = node[2]
            break
                    
        if(node[1] not in visited):
            for i in range(1, len(cost[node[1]])):
                if(cost[node[1]][i]!=-1):
                    heapq.heappush(heap, (node[3] + cost[node[1]][i] + heuristic[i], i, node[2]+[i], node[3] + cost[node[1]][i]))
        visited.append(node[1])
    return l

def UCS_Traversal(cost, start_point, goals):
    l = []
    heap = []
    heapq.heappush(heap, (0, start_point, [start_point]))
    visited = []
    while(heap):
        node = heapq.heappop(heap)
        if node[1] in goals:
            l = node[2]
            break

        if(node[1] not in visited):
            for i in range(1, len(cost[node[1]])):
                if(cost[node[1]][i]>0):
                    heapq.heappush(heap, (node[0]+cost[node[1]][i], i, node[2]+[i]))
        visited.append(node[1])
    return l

def rec_DFS(cost,start_point,goals,visited,n,soln):
    visited[start_point-1] = 1
    if start_point in goals:
        return soln
    
    for i in range(1,n+1):
        if cost[start_point][i]!=0 and cost[start_point][i]!=-1 and visited[i-1]==0:
            old_len = len(soln)
            soln.append(i)
            soln = rec_DFS(cost,i,goals,visited,n,soln)
            if(len(soln)!=old_len):
                return soln

    if(start_point in soln):
        soln.remove(start_point)
                
    return soln



def DFS_Traversal(cost,start_point,goals):

    l = []
    visited = []
    n = len(cost[0]) - 1
    visited.extend([0]*n)
    l.append(start_point)
    l = rec_DFS(cost,start_point,goals,visited,n,l)
    return l


'''
Function tri_traversal - performs DFS, UCS and A* traversals and returns the path for each of these traversals 

n - Number of nodes in the graph
m - Number of goals ( Can be more than 1)
1<=m<=n
Cost - Cost matrix for the graph of size (n+1)x(n+1)
IMP : The 0th row and 0th column is not considered as the starting index is from 1 and not 0. 
Refer the sample test case to understand this better

Heuristic - Heuristic list for the graph of size 'n+1' 
IMP : Ignore 0th index as nodes start from index value of 1
Refer the sample test case to understand this better

start_point - single start node
goals - list of size 'm' containing 'm' goals to reach from start_point

Return : A list containing a list of all traversals [[],[],[]]
1<=m<=n
cost[n][n] , heuristic[n][n], start_point, goals[m]

NOTE : you are allowed to write other helper functions that you can call in the given fucntion
'''

def tri_traversal(cost, heuristic, start_point, goals):
    l = []

    t1 = DFS_Traversal(cost, start_point, goals)
    t2 = UCS_Traversal(cost, start_point, goals)
    t3 = A_star_Traversal(cost, heuristic, start_point, goals)

    l.append(t1)
    l.append(t2)
    l.append(t3)
    return l

