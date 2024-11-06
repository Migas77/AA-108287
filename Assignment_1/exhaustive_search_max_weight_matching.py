import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

from random_graph_generator import generate_random_graph


def exhaustive_search_max_weight_matching(G):
    """
    Find the maximum weighted matching using exhaustive search.

    :param G: A networkx.Graph instance with weighted edges.
    :return:
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
    n_nodes = 12
    edge_density = 0.25
    Graph = generate_random_graph(n_nodes, edge_density)
    expected_result = nx.max_weight_matching(Graph)
    print(
        expected_result,
        sum(Graph.get_edge_data(*e)["weight"] for e in expected_result)
    )
    print(exhaustive_search_max_weight_matching(Graph))

    pos = nx.get_node_attributes(Graph, 'pos')
    labels = nx.get_edge_attributes(Graph, 'weight')
    nx.draw(Graph, pos, with_labels=True)
    nx.draw_networkx_edge_labels(Graph, pos, edge_labels=labels)
    plt.plot()
    plt.show()

# Edmond Blossom Algorithm
# networkX max_weight_matching
