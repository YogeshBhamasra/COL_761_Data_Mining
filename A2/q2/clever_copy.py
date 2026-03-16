import argparse as ap
import time
import networkx as nx

def get_seeds(seed_path):
    seeds = set()
    with open(seed_path, "r") as f:
        for line in f:
            seeds.add(line.strip())
    return seeds

def get_edges(dataset_path):
    edges = []
    probs = {}
    with open(dataset_path, "r") as f:
        for line in f:
            u, v, p = line.strip().split()
            edges.append((u, v, float(p)))
            probs[(u, v)] = float(p)
    return edges, probs
    
def get_burnable_edges(G, node, hops):
    burnable_edges = set()
    for neighbor in G.neighbors(node):
        burnable_edges.add((node, neighbor))
        if hops > 1:
            burnable_edges.update(get_burnable_edges(G, neighbor, hops - 1))
    return burnable_edges

def make_graph(edges, seeds, hops=-1):
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)
    if hops > 0:
        burnable_edgses = set()
        for seed in seeds:
            burnable_edgses.update(get_burnable_edges(G, seed, hops))
        G = G.edge_subgraph(burnable_edgses).copy()
    return G

def get_removable_edges(G, seeds, k):
    removable_edges = set()
    for seed in seeds:
        for neighbor in G.neighbors(seed):
            removable_edges.add((seed, neighbor))
    print(f"Total burnable edges from seeds: {len(removable_edges)}")
    return sorted(removable_edges, key=lambda x: G[x[0]][x[1]]['weight'], reverse=True)[:k]
    


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
    
    
    edges_to_remove = get_removable_edges(G, seeds, args.k)
    
    print(f"Number of Edges before removal: {G.number_of_edges()}")
    # remove_edges(G, edges_to_remove)
    G.remove_edges_from(edges_to_remove)
    print(f"Number of Edges after removal: {G.number_of_edges()}")
    
    with open(args.output_path, "w") as f:
        for u, v in edges_to_remove:
            f.write(f"{u} {v}\n")