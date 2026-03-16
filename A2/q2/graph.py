import networkx as nx
import matplotlib.pyplot as plt


def get_seeds(file_path):
    with open(file_path, "r") as f:
        seeds = set()
        for line in f:
            seed = line.strip()
            seeds.add(seed)
    return seeds


def get_edges(file_path):

    with open(file_path, "r") as f:
        nodes = set()
        edges = []
        num = 0
        for line in f:
            num += 1
            u, v, p = line.strip().split()
            nodes.add(u)
            nodes.add(v)
            edges.append((u, v, float(p)))

            # if num == 10:
            #     break

        probs = {node: {} for node in nodes}

        for u, v, p in edges:
            probs[u][v] = p
    return edges, probs

def get_burnable_edges(G, node, hops):
    burnable_edges = set()
    for neighbor in G.neighbors(node):
        burnable_edges.add((node, neighbor))
        if hops > 1:
            burnable_edges.update(get_burnable_edges(G, neighbor, hops - 1))
    return sorted(burnable_edges)
    
def remove_edges(G, edges_to_remove):
    for u, v in edges_to_remove:
        # print(f"Removing edge: ({u}, {v})")
        if G.has_edge(u, v):
            G.remove_edge(u, v)
            # print(f"Removed edge: ({u}, {v})")

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
