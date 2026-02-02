from utils import parse_graph
import numpy as np
import argparse
import networkx as nx 
from networkx.algorithms import isomorphism

def isomorphic(G, S):
    def node_match(n1, n2):
        return n1.get('label') == n2.get('label')
    
    # Edge match function: edges match if labels are equal
    def edge_match(e1, e2):
        return e1.get('label') == e2.get('label')
    
    # Create graph matcher
    matcher = isomorphism.GraphMatcher(
        G, 
        S,
        node_match=node_match,
        edge_match=edge_match
    )
    
    # Check if subgraph isomorphism exists
    return matcher.subgraph_is_isomorphic()

def make_networkx_graph(graph):
    G = nx.Graph()
    for info in graph: 
        if len(info) == 2:
            G.add_node(info[0], label=info[1])
        elif len(info) == 3:
            G.add_edge(info[0], info[1], label=info[2])
    return G

def feature_matrix(graphs, subgraphs):
    feature_matrix = np.zeros((len(graphs), len(subgraphs)), dtype=int)

    subgraph_nx = [make_networkx_graph(subgraph) for subgraph in subgraphs]
    for i, graph in enumerate(graphs):
        G = make_networkx_graph(graph)
        for j, subgraph in enumerate(subgraph_nx):
            if isomorphic(G, subgraph):
                feature_matrix[i, j] = 1

    return feature_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse a graph from a file and print its adjacency list.")
    parser.add_argument("graph_path", type=str, help="Path to the graph file")
    parser.add_argument("subgraph_path", type=str, help="Path to the subgraph file")
    parser.add_argument("path_features", type=str, help="Path to save the feature matrix")
    args = parser.parse_args()

    graph = parse_graph(args.graph_path)
    subgraph = parse_graph(args.subgraph_path)

    feature_mat = feature_matrix(graph, subgraph)
    np.save(args.path_features, feature_mat)


        