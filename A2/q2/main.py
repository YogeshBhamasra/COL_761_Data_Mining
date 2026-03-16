import argparse as ap
import graph
import cluster
import simulation
import clever
import copy

def get_arguments():
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
    return argparse.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    edges, probs = graph.get_edges(args.dataset_path)
    seeds = graph.get_seeds(args.seed_path)
    G = graph.make_graph(edges, seeds, hops=args.hops)
    
    burnt_original = simulation.simulate(G, seeds, probs, args.num_sims)
    
    G_clustered = copy.deepcopy(G)
    edges_to_remove_cluster = cluster.modify_graph(G_clustered, seeds, probs, args.k)
    burnt_after_removal_clustered = simulation.simulate(G_clustered, seeds, probs, args.num_sims)
    
    G_clever = copy.deepcopy(G)
    edges_to_remove_clever = clever.modify_graph(G_clever, seeds, args.k)
    burnt_after_removal_clever = simulation.simulate(G_clever, seeds, probs, args.num_sims)
    
    # print(f"Original burnt nodes: {burnt_original}")
    # print(f"Burnt nodes after clustered removal: {burnt_after_removal_clustered}")
    # print(f"Burnt nodes after clever removal: {burnt_after_removal_clever}")
    if burnt_after_removal_clustered < burnt_after_removal_clever:
        edges_to_remove = edges_to_remove_cluster
        G_final = G_clustered
    elif burnt_after_removal_clustered > burnt_after_removal_clever:
        edges_to_remove = edges_to_remove_clever
        G_final = G_clever
    else:
        edges_to_remove = edges_to_remove_cluster
        G_final = G_clustered
    
    if len(edges_to_remove) < args.k:
        edges = sorted(G_final.edges())
        
        for u, v in sorted(edges, key=lambda x: G_final[x[0]][x[1]]['weight'], reverse=True):
            if (u, v) not in edges_to_remove:
                edges_to_remove.append((u, v))
                if len(edges_to_remove) == args.k:
                    break
        
    with open(args.output_path, "w") as f:
        for u, v in edges_to_remove:
            f.write(f"{u} {v}\n")
