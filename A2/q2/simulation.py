import random

def simulate_burning(graph, seeds, probs, rng):
    burning = set(seeds)
    burnt = set()
    # burnable_edges = set()
    
    # if hops != -1:
    #     for seed in seeds:
    #         burnable_edges.update(get_burnable_edges(graph, seed, hops))
    
    while True:
        new_burning = set()
        for node in sorted(list(burning)):
            for neighbor in sorted(list(graph.neighbors(node))):
                if neighbor not in burnt and neighbor not in burning:
                    p = probs[node].get(neighbor, 0)
                    if rng.random() < p:
                        new_burning.add(neighbor)
        burnt.update(burning)
        burning = new_burning
        if not new_burning:
            break
    return burnt
    
def simulate(graph, seeds, probs, num_sims, base_seed=42):    
    total_burnt = 0
    rng = random.Random(base_seed)
    for _ in range(num_sims):
        burnt = simulate_burning(graph, seeds, probs, rng)
        total_burnt += len(burnt)
    return total_burnt / num_sims


