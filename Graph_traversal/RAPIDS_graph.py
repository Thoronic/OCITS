import cugraph
import networkx as nx
import time

input_filename = "Graph_traversal/synthetic_graph.csv"
graph = nx.read_edgelist(input_filename, delimiter=";", nodetype=int, data=False)

source_vertex_id = 0 
num_iterations = 20

# Warm-up the GPU
result = cugraph.bfs(graph, start=source_vertex_id)
result.to_csv("Graph_traversal/result.csv")

# Measure the time for BFS on GPU
time_var = 0
for i in range(num_iterations):
    start_time = time.time()
    result = cugraph.bfs(graph, start=source_vertex_id)
    end_time = time.time()
    time_var += end_time - start_time
    result.to_csv("Graph_traversal/result.csv")

# Calculate average time per iteration
print(result)
average_time = time_var / num_iterations

print(f"Average time for graph traversal: {average_time} seconds")