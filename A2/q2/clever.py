import graph

def get_removable_edges(G, seeds, k):
    removable_edges = set()
    for seed in seeds:
        for neighbor in G.neighbors(seed):
            removable_edges.add((seed, neighbor))
    return sorted(sorted(removable_edges), key=lambda x: G[x[0]][x[1]]['weight'], reverse=True)[:k]
    

def modify_graph(G, seeds, k):
    removable_edges = get_removable_edges(G, seeds, k)
    graph.remove_edges(G, removable_edges)
    return removable_edges