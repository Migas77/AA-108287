import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

from random_graph_generator import generate_random_graph


def greedy_heuristic_max_weight_matching(G):
    max_weight = 0
    max_weight_matching = []

    # Sort edges by vertex degree (for each of the vertex of the the edge) - in ascending order
    sorted_edges = sorted(G.edges(data="weight"), key=lambda x: G.degree(x[0]) + G.degree(x[1]))

    # Sort edges by weight in descending order
    # sorted_edges = sorted(G.edges(data="weight"), key=lambda x: x[2], reverse=True)

    # Sort edges by weight in descending order - prefer edges with vertices with lower degree
    sorted_edges = sorted(G.edges(data="weight"), key=lambda x: x[2] / (G.degree(x[0]) + G.degree(x[1]) + 1), reverse=True)

    # Get all edges with weights
    while len(sorted_edges) > 0:
        edge = sorted_edges.pop(0)
        max_weight_matching.append(edge)
        max_weight += edge[2]
        sorted_edges = [e for e in sorted_edges if e[0] not in edge[:2] and e[1] not in edge[:2]]

    return max_weight_matching, max_weight


if __name__ == '__main__':
    n_nodes = 19
    edge_density = 0.25
    count = 0
    for n_nodes in list(range(10, 20)):
        for edge_density in [0.125, 0.25, 0.5, 0.75]:
            print(f"n_nodes: {n_nodes}, edge_density: {edge_density}")
            Graph = generate_random_graph(n_nodes, edge_density)
            expected_result = nx.max_weight_matching(Graph)
            print(
                expected_result,
                sum(Graph.get_edge_data(*e)["weight"] for e in expected_result)
            )
            result = greedy_heuristic_max_weight_matching(Graph)
            if sum(Graph.get_edge_data(*e)["weight"] for e in expected_result) == result[1]:
                count += 1
    print("count = ", count)

    # pos = nx.get_node_attributes(Graph, 'pos')
    # labels = nx.get_edge_attributes(Graph, 'weight')
    # nx.draw(Graph, pos, with_labels=True)
    # nx.draw_networkx_edge_labels(Graph, pos, edge_labels=labels)
    # plt.plot()
    # plt.show()

# Edmond Blossom Algorithm
# networkX max_weight_matching
