import tensorflow as tf
import networkx as nx
import numpy as np
import time

def bfs(graph, start_node):
    with tf.device("/device:GPU:0"):
        num_nodes = tf.shape(graph)[0]
        visited = tf.fill((num_nodes,), False)  # Mark all nodes as not visited
        queue = tf.constant([start_node])  # Initialize the queue with the start node
        start_time = time.time()
        visited = tf.tensor_scatter_nd_update(visited, [[start_node]], [True])  # Mark start node as visited

        while tf.size(queue) > 0:
            node = queue[0]  # Dequeue
            queue = queue[1:]  # Remove the first element

            # Process neighbors of the current node
            neighbors = tf.where(graph[node] > 0)[:, 0]  # Find neighbors of the current node
            for neighbor in neighbors:
                if not visited[neighbor]:
                    queue = tf.concat([queue, [neighbor]], axis=0)  # Enqueue
                    visited = tf.tensor_scatter_nd_update(visited, [[neighbor]], [True])  # Mark neighbor as visited
        end_time = time.time()
    return end_time - start_time


input_filename = "Graph_traversal/synthetic_graph.csv"
num_iterations = 20
source_vertex_id = 0 
graph = nx.read_edgelist(input_filename, delimiter=";", nodetype=int, data=False)


adjacency_matrix = nx.adjacency_matrix(graph)
adjacency_matrix = adjacency_matrix.toarray()
adjacency_tensor = tf.constant(np.array(adjacency_matrix))

# Warm-up the GPU
bfs(adjacency_tensor, source_vertex_id)

time_var = 0
for i in range(num_iterations):
    time_var += bfs(adjacency_tensor, source_vertex_id)

average_time = time_var / num_iterations

print(f"Average time for graph traversal: {average_time} seconds")