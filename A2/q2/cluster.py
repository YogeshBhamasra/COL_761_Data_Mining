import networkx as nx
# import numpy as np
import markov_clustering as mc
import time
import score
import graph


def get_bridge_edges_bw(cluster1, cluster2, graph):
    bridge_edges = []
    for u in cluster1:
        for v in cluster2:
            if graph.has_edge(u, v):
                bridge_edges.append((u, v))
            if graph.has_edge(v, u):
                bridge_edges.append((v, u))
    return bridge_edges

    
def get_bridge_edges(clusters, graph):
    node_to_cluster = {}
    for i, cluster in enumerate(clusters):
        for node in cluster:
            node_to_cluster[node] = i
    bridge_edges = []
    for u, v in graph.edges():
        if node_to_cluster[u] != node_to_cluster[v]:
            bridge_edges.append((u, v))
    return bridge_edges

def get_edges_to_remove(scores, k):
    sorted_scores = sorted(sorted(scores.items()), key=lambda x: x[1], reverse=True)
    top_k_scores = sorted_scores[:k]
    
    top_k_edges = [edge for edge, score in top_k_scores]
    return top_k_edges

def cluster(G):
    
    # Get the adjacency matrix of the graph
    adjacency_matrix = nx.to_numpy_array(G, weight='weight')
    
    # print("Adjacency Matrix:")
    # print(adjacency_matrix)
    
    print("Running Markov Clustering...")
    # Perform Markov Clustering
    result = mc.run_mcl(adjacency_matrix, inflation=2, verbose=False)
    
    # Get the clusters
    clusters = mc.get_clusters(result)
    
    nodes = list(G.nodes())
    
    # Get clusters as lists of node labels
    clusters = [[nodes[i] for i in cluster] for cluster in clusters]
    
    return clusters
    
def modify_graph(G, seeds, probs, k):
    start_time = time.time()
    clusters = cluster(G)
    end_time = time.time()

    bridge_edges = get_bridge_edges(clusters, G)

    ppr = score.personalized_pagerank(G, seeds)
    scores = score.get_scores(ppr, bridge_edges, probs)
    
    edges_to_remove = get_edges_to_remove(scores, k)
    graph.remove_edges(G, edges_to_remove)
    return edges_to_remove
    