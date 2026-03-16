import networkx as nx

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
