import re
from cluster import cluster
import argparse as ap
import networkx as nx
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from tqdm import tqdm
import time



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

            # if num == 10:
            #     break

        probs = {node: {} for node in nodes}

        for u, v, p in edges:
            probs[u][v] = p
        print(f"No.of nodes : {len(nodes)}")
    return edges, probs


def make_graph(edges):
    G = nx.DiGraph()
    for u, v, p in edges:
        G.add_edge(u, v, weight=p)
    # G.add_weighted_edges_from(edges)
    
    # pos = nx.spring_layout(G, seed=7)
    
    # nx.draw_networkx(G, pos, with_labels=True,
    #                  node_color='lightblue',
    #                  edge_color='gray')
    
    # extract weights
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    
    # print(edge_labels)
    
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos = 0.3)
    
    # plt.show()
    return G


def get_seeds(file_path):
    with open(file_path, "r") as f:
        seeds = set()
        for line in f:
            seed = line.strip()
            seeds.add(seed)
    return seeds


def get_bridge_edges_bw(cluster1, cluster2, graph):
    bridge_edges = []
    for u in cluster1:
        for v in cluster2:
            if graph.has_edge(u, v):
                bridge_edges.append((u, v))
            if graph.has_edge(v, u):
                bridge_edges.append((v, u))
    return bridge_edges

# def get_bridge_edges(clusters, graph):
#     bridge_edges = []
#     # for i in range(len(clusters)):
#     #     for j in range(i + 1, len(clusters)):
#     #         for u in clusters[i]:
#     #             for v in clusters[j]:
#     #                 if graph.has_edge(u, v):
#     #                     bridge_edges.append((u, v))
#     #                 if graph.has_edge(v, u):
#     #                     bridge_edges.append((v, u))
#     # 
    
#     with tqdm_joblib(tqdm(desc="Finding bridge edges", total=len(clusters) * (len(clusters) - 1) // 2)):
#         results = Parallel(n_jobs=-1)(
#             delayed(get_bridge_edges_bw)(clusters[i], clusters[j], graph)
#             for i in range(len(clusters))
#             for j in range(i + 1, len(clusters))
#         )
    
#     for res in results:
#         bridge_edges.extend(res)
                        
#     return bridge_edges
    
def get_bridge_edges(clusters, graph):
    node_to_cluster = {}
    for i, cluster in tqdm(enumerate(clusters)):
        for node in cluster:
            node_to_cluster[node] = i
    bridge_edges = []
    for u, v in tqdm(graph.edges(), desc="Finding bridge edges"):
        if node_to_cluster[u] != node_to_cluster[v]:
            bridge_edges.append((u, v))
    return bridge_edges


def personalized_pagerank(graph, seeds, alpha=0.85, max_iter=100):
    v = {node: 0.0 for node in graph.nodes()}
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
    argparse = ap.ArgumentParser(
        description="Program to reduce the number of burning nodes by removing k edges"
    )
    argparse.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset file containing edges in the format: u v p",
    )
    argparse.add_argument(
        "--seed_path",
        type=str,
        required=True,
        help="Path to the seed file containing initial burning nodes",
    )
    argparse.add_argument(
        "--k", type=int, required=True, help="Number of edges to remove"
    )
    argparse.add_argument(
        "--num_sims",
        type=int,
        required=True,
        help="Number of simulations to run for averaging results",
    )
    args = argparse.parse_args()
    edges, probs = get_edges(args.dataset_path)
    G = make_graph(edges)

    values = []
    for i in range(args.num_sims):
        print(f"Simulation {i + 1}/{args.num_sims}")
        burnt_original = simulate_burning(G, get_seeds(args.seed_path), probs)
        print(f"Number of burnt nodes before edge removal: {len(burnt_original)}")
        values.append(len(burnt_original))

    print(
        f"Average number of burnt nodes before edge removal: {sum(values) / len(values)}"
    )
    
    start_time = time.time()
    clusters = cluster(G)
    end_time = time.time()
    print(f"Clustering completed in {((end_time - start_time) / 60):.2f} minutes")
    print(f"Number of clusters: {len(clusters)}")

    bridge_edges = get_bridge_edges(clusters, G)

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
        print(f"Simulation {i + 1}/{args.num_sims} after edge removal")
        burnt_after_removal = simulate_burning(G, seeds, probs)
        print(f"Number of burnt nodes after edge removal: {len(burnt_after_removal)}")
        values.append(len(burnt_after_removal))
    print(
        f"Average number of burnt nodes after edge removal: {sum(values) / len(values)}"
    )
