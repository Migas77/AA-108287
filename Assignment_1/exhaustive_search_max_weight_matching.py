import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations


def exhaustive_search_max_weight_matching(G):
    """
    Find the maximum weighted matching using exhaustive search.

    Parameters:
    - G: A networkx.Graph instance with weighted edges.

    Returns:
    - max_weight_matching: Set of edges in the maximum weighted matching.
    - max_weight: The total weight of the maximum weighted matching.
    """
    max_weight = 0
    max_weight_matching = []

    # Get all edges with weights
    edges = G.edges(data="weight")
    # Generate all subsets of edges (with size 1, 2, 3, ..., len(edges))
    for r in range(1, len(edges) + 1):
        for subset in combinations(edges, r):
            # Extract edge pairs
            # no need to extract the weights as I'm using nx.is_matching
            edge_set = {(u, v) for u, v, _ in subset}

            # Check if the subset is a valid matching
            if nx.is_matching(G, edge_set):
                # Calculate the total weight of the matching
                # and update if found maximum weight
                weight = sum(weight for _, _, weight in subset)
                if weight > max_weight:
                    max_weight = weight
                    max_weight_matching = edge_set

    return max_weight_matching, max_weight


if __name__ == '__main__':
    Graph = nx.Graph()
    Graph.add_weighted_edges_from(
        [(1, 2, 2), (1, 3, 1), (2, 3, 4), (2, 4, 3)]
    )
    pos = nx.spring_layout(Graph, seed=7)
    labels = nx.get_edge_attributes(Graph, 'weight')
    nx.draw(
        Graph, pos, with_labels=True
    )
    nx.draw_networkx_edge_labels(Graph, pos, edge_labels=labels)
    plt.plot()
    plt.show()
    print(
        nx.max_weight_matching(Graph),
        sum(Graph.get_edge_data(*e)["weight"] for e in nx.max_weight_matching(Graph))
    )
    print(exhaustive_search_max_weight_matching(Graph))

# Edmond Blossom Algorithm
# networkX max_weight_matching
