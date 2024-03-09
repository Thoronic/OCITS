import networkx as nx

def generate_synthetic_graph(nodes, edges):
    G = nx.gnm_random_graph(nodes, edges)
    return G

def save_graph_to_file(graph, filename):
    nx.write_edgelist(graph, filename, delimiter=";", data=False)

if __name__ == "__main__":
    num_nodes = 20000
    num_edges = 40000
    output_filename = "Graph_traversal/synthetic_graph.csv"

    synthetic_graph = generate_synthetic_graph(num_nodes, num_edges)
    save_graph_to_file(synthetic_graph, output_filename)