from tkinter.constants import TOP
from cluster import cluster
import argparse as ap
import networkx as nx
import random
from tqdm import tqdm
import time
from matplotlib import pyplot as plt



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


def make_graph(edges, seeds, hops=-1):
    G = nx.DiGraph()
    for u, v, p in edges:
        G.add_edge(u, v, weight=p)
    
    if hops == -1:
        return G
        
    G_hops = nx.DiGraph()
    # addd edges within hops of seeds
    # for seed in seeds:
    #     for node in nx.single_source_shortest_path_length(G, seed, cutoff=hops).keys():
    #         G_hops.add_node(node)
    # for u, v, p in edges:
    #     if G_hops.has_node(u) and G_hops.has_node(v):
    #         G_hops.add_edge(u, v, weight=p)
    # return G_hops
    # 
    
    burnable_edges = set()
    for seed in seeds:
        burnable_edges.update(get_burnable_edges(G, seed, hops))
        
    for u, v, p in edges:
        if (u, v) in burnable_edges or (v, u) in burnable_edges:
            G_hops.add_edge(u, v, weight=p)
    G = G_hops
    
    # pos = nx.spring_layout(G, seed=7)
    
    # nx.draw_networkx(G, pos, with_labels=True,
    #                  node_color='lightblue',
    #                  edge_color='gray')
    
    # # extract weights
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    
    # # print(edge_labels)
    
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
        score = pagerank[u] * probs[u].get(v, 0) / pagerank[v]
        # score = pagerank[u] * probs[u].get(v, 0) - pagerank[v]
        # score = pagerank[u] * probs[u].get(v, 0)
        scores[(u, v)] = score
    return scores


def get_edges_to_remove(scores, k):
    print(f"Getting {k} edges to remove based on scores...")
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_k_scores = sorted_scores[:k]
    
    top_k_edges = [edge for edge, score in top_k_scores]
    print(f"Selected edges to remove: {top_k_edges}")
    return top_k_edges


def remove_edges(graph, edges_to_remove):
    print(f"Removing {len(edges_to_remove)} edges...")
    for u, v in edges_to_remove:
        # print(f"Removing edge: ({u}, {v})")
        if graph.has_edge(u, v):
            graph.remove_edge(u, v)
            # print(f"Removed edge: ({u}, {v})")

def get_burnable_edges(graph, node, hops):
    burnable_edges = set()
    for neighbor in graph.neighbors(node):
        burnable_edges.add((node, neighbor))
        if hops > 1:
            burnable_edges.update(get_burnable_edges(graph, neighbor, hops - 1))
    return burnable_edges

def simulate_burning(graph, seeds, probs):
    burning = set(seeds)
    burnt = set()
    # burnable_edges = set()
    
    # if hops != -1:
    #     for seed in seeds:
    #         burnable_edges.update(get_burnable_edges(graph, seed, hops))
    
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
    
    argparse.add_argument(
        "--hops",
        type=int,
        required=True,
        help="Number of hops to consider for the graph (default: -1 for full graph)",
    )
    
    argparse.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the blocked edges"
    )
    args = argparse.parse_args()
    edges, probs = get_edges(args.dataset_path)
    seeds = get_seeds(args.seed_path)
    G = make_graph(edges, seeds, hops=args.hops)

    # values = []
    # for i in range(args.num_sims):
    #     print(f"Simulation {i + 1}/{args.num_sims}")
    #     burnt_original = simulate_burning(G, seeds, probs)
    #     print(f"Number of burnt nodes before edge removal: {len(burnt_original)}")
    #     values.append(len(burnt_original))

    # print(
    #     f"Average number of burnt nodes before edge removal: {sum(values) / len(values)}"
    # )
    
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
    
    print(f"Number of Edges before removal: {G.number_of_edges()}")
    edges_to_remove = get_edges_to_remove(scores, args.k)

    remove_edges(G, edges_to_remove)
    print(f"Number of Edges after removal: {G.number_of_edges()}")
    
    with open(args.output_path, "w") as f:
        for u, v in edges_to_remove:
            f.write(f"{u} {v}\n")

    # values = []
    # for i in range(args.num_sims):
    #     print(f"Simulation {i + 1}/{args.num_sims} after edge removal")
    #     burnt_after_removal = simulate_burning(G, seeds, probs)
    #     print(f"Number of burnt nodes after edge removal: {len(burnt_after_removal)}")
    #     values.append(len(burnt_after_removal))
    # print(
    #     f"Average number of burnt nodes after edge removal: {sum(values) / len(values)}"
    # )
