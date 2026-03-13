import networkx as nx
# import numpy as np
import markov_clustering as mc

def cluster(G):
    
    # Get the adjacency matrix of the graph
    adjacency_matrix = nx.to_numpy_array(G, weight='weight')
    
    # print("Adjacency Matrix:")
    # print(adjacency_matrix)
    
    print("Running Markov Clustering...")
    # Perform Markov Clustering
    result = mc.run_mcl(adjacency_matrix, inflation=1.2, verbose=True)
    
    # Get the clusters
    clusters = mc.get_clusters(result)
    
    nodes = list(G.nodes())
    
    # Get clusters as lists of node labels
    clusters = [[nodes[i] for i in cluster] for cluster in clusters]
    
    return clusters
    