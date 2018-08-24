
# Depth-First Search(DFS), Topological Sort

adjacency lists:  
![](https://latex.codecogs.com/gif.latex?Adj%5Bu%5D%20%3D%20%5Cleft%20%5C%7B%20v%5Cin%20V%20%7C%20%28u%2Cv%29%20%5Cin%20E%5Cright%20%5C%7D)     
BFS: visit the vertices layer by layer, and could find the shortest path easily, but only visit unreachable vertices.  
DFS: explore the whole graph.  


## Depth-First Search
* recursively explore graph, backtracking as necessary.
* careful not to revisit
```
#recursive part
parent = {s: None}
def DFS-Visit(V, Adj, s):
  for v in Adj[s]:
    if v not in parent:
      parent[v]=s
      DFS-Visit(V, Adj, v)

#algorithm
def DFS(V, Adj):
  parent = {}
  for s in V:
    if s not in parent:
      parent[s]=None
      DFS-Visit(V,Adj,s)
```
**Analysis**: O(V+E)(linear time)  
* visit each vertex once in DFS alone: O(V)  
* DFS-Visit(V, Adj, v) called once per vertex v  
=> ![](https://latex.codecogs.com/gif.latex?O%28%5Csum_%7Bv%5Cin%20V%7D%7CAdj%5Bv%5D%7C%20%29%20%3D%20O%28E%29).  

