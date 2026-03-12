import re
from cluster import cluster
import argparse as ap
import networkx as nx
import random

def get_edges(file_path):
    
    with open(file_path, "r") as f:
        nodes = set()
        edges = []
        num = 0
        for line in f:
            num += 1
            u, v, p = line.strip().split()
            if u == 0 or v == 0:
                print(f"Got zero node: {u} or {v}")
            nodes.add(u)
            nodes.add(v)
            edges.append((u, v, float(p)))
            
            # if num == 1000:
            #     break
        
        probs = {node: {} for node in nodes}
        
        for u, v, p in edges:
            probs[u][v] = p
        print(f"No.of nodes : {len(nodes)}")
    return edges, probs

def make_graph(edges):
    G = nx.Graph()
    for u, v, p in edges:
        G.add_edge(u, v, weight=p)
    return G

def get_seeds(file_path):
    with open(file_path, "r") as f:
        seeds = set()
        for line in f:
            seed = line.strip()
            seeds.add(seed)
    return seeds
    
def get_bridge_edges(clusters):
    bridge_edges = []
    for i in range(len(clusters)):
        for j in range(i+1, len(clusters)):
            for u in clusters[i]:
                for v in clusters[j]:
                    bridge_edges.append((u, v))
    return bridge_edges 
 
def personalized_pagerank(graph, seeds, alpha=0.85, max_iter=100):
    v = {node: 0. for node in graph.nodes()}
    for seed in seeds:
        v[seed] = 1 / len(seeds)
    
    pagerank = nx.pagerank(graph, alpha=alpha, personalization=v, max_iter=max_iter)
    return pagerank

def get_scores(pagerank, bridge_edges, probs):
    scores = {}
    for u, v in bridge_edges:
        score = pagerank[u] * probs[u].get(v, 0)
        scores[(u, v)] = score
    return scores
    
def get_edges_to_remove(scores, k):
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores[:k]
    
def remove_edges(graph, edges_to_remove):
    for u, v in edges_to_remove:
        if graph.has_edge(u, v):
            graph.remove_edge(u, v)
            

def simulate_burning(graph, seeds, probs):
    burning = set(seeds)
    burnt = set()
    while True:
        new_burning = set()
        for node in burning:
            for neighbor in graph.neighbors(node):
                if neighbor not in burnt and neighbor not in burning:
                    p = probs[node].get(neighbor, 0)
                    if random.random() < p:
                        new_burning.add(neighbor)
                        
        if not new_burning:
            break
        burnt.update(burning)
        burning = new_burning
    return burnt
    
if __name__ == "__main__":
    argparse = ap.ArgumentParser(description="Program to reduce the number of burning nodes by removing k edges")
    argparse.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset file containing edges in the format: u v p")
    argparse.add_argument("--seed_path", type=str, required=True, help="Path to the seed file containing initial burning nodes")
    argparse.add_argument("--k", type=int, required=True, help="Number of edges to remove")
    argparse.add_argument("--num_sims", type=int, required=True, help="Number of simulations to run for averaging results")
    args = argparse.parse_args()
    edges, probs = get_edges(args.dataset_path)
    G = make_graph(edges)
    
    values = []
    for i in range(args.num_sims):
        print(f"Simulation {i+1}/{args.num_sims}")
        burnt_original = simulate_burning(G, get_seeds(args.seed_path), probs)
        print(f"Number of burnt nodes before edge removal: {len(burnt_original)}")
        values.append(len(burnt_original))
    
    print(f"Average number of burnt nodes before edge removal: {sum(values) / len(values)}")

    clusters = cluster(G)
    print(f"Number of clusters: {len(clusters)}")
    
    bridge_edges = get_bridge_edges(clusters)
    
    print(f"Number of bridge edges: {len(bridge_edges)}")
    seeds = get_seeds(args.seed_path)
    print(f"Number of seeds: {len(seeds)}")
    ppr = personalized_pagerank(G, seeds)
    print("Personalized PageRank calculated.")
    scores = get_scores(ppr, bridge_edges, probs)
    
    edges_to_remove = get_edges_to_remove(scores, args.k)
    
    remove_edges(G, edges_to_remove)
    
    values = []
    for i in range(args.num_sims):
        print(f"Simulation {i+1}/{args.num_sims} after edge removal")
        burnt_after_removal = simulate_burning(G, seeds, probs)
        print(f"Number of burnt nodes after edge removal: {len(burnt_after_removal)}")
        values.append(len(burnt_after_removal))
    print(f"Average number of burnt nodes after edge removal: {sum(values) / len(values)}")