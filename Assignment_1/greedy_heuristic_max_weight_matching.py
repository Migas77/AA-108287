from itertools import combinations
from collections import defaultdict
from tabulate import tabulate
import pickle
import time
import networkx as nx
import matplotlib.pyplot as plt


# ver ordem de complexidade de sort
# não vai aparecer operações básicas disto

def greedy_heuristic_vertices_degrees_max_weight_matching(G):
    # Sort edges by vertex degree (for each of the vertex of the the edge) - in ascending order
    return greedy_heuristic_max_weight_matching2(G, lambda x: G.degree(x[0]) + G.degree(x[1]))

def greedy_heuristic_edges_weight_max_weight_matching(G):
    # Sort edges by weight in descending order
    return greedy_heuristic_max_weight_matching2(G, lambda x: x[2], reverse=True)

def greedy_heuristic_mix_max_weight_matching(G):
    # Sort edges by weight in descending order, but prefer edges with vertices with lower degree
    return greedy_heuristic_max_weight_matching2(G, lambda x: x[2] / (G.degree(x[0]) + G.degree(x[1]) + 1), reverse=True)

    
def greedy_heuristic_max_weight_matching2(G, criteria, reverse=False):
    max_weight = 0
    matched_vertices = set()
    max_weight_matching = []

    sorted_edges = sorted(G.edges(data="weight"), key=criteria, reverse=reverse)

    # Get all edges with weights
    for edge in sorted_edges:
        u, v, weight = edge
        if u not in matched_vertices and v not in matched_vertices:
            matched_vertices.add(u)
            matched_vertices.add(v)
            max_weight_matching.append(edge)
            max_weight += weight
            
    return max_weight_matching, max_weight


def greedy_heuristic_max_weight_matching(G, criteria, reverse=False):
    max_weight = 0
    max_weight_matching = []

    sorted_edges = sorted(G.edges(data="weight"), key=criteria, reverse=reverse)

    # Get all edges with weights
    while len(sorted_edges) > 0:
        edge = sorted_edges.pop(0)
        max_weight_matching.append(edge)
        max_weight += edge[2]
        sorted_edges = [e for e in sorted_edges if e[0] not in edge[:2] and e[1] not in edge[:2]]

    return max_weight_matching, max_weight


if __name__ == '__main__':
    subsets_counter = 0
    is_matching_counter = 0
    headers = ['#Nodes', 'Edge Density', '#Edges (n)', '#Basic Operations', 'Execution Time (s)', '#Configs Tested', 'isExpectedResult']
    results = [] 
    num_subset_by_edge_density_by_nodes = defaultdict(defaultdict)
    num_ismatching_by_edge_density_by_nodes = defaultdict(defaultdict)
    execution_time_by_edge_density_by_nodes = defaultdict(defaultdict)
    """
    num_vertices_4: {
        densidade_0.125: [resultado1, resultado2, ...]
    }
    """
    for n_nodes in range(4, 100):
        for edge_density in [0.125, 0.25, 0.5, 0.75]:
            is_matching_counter = 0
            subsets_counter = 0
            edge_density_str = str(edge_density).replace('.', '')
            with open(f"generated_graphs/graph_{n_nodes}_{edge_density_str}.gpickle", "rb") as f:
                Graph = pickle.load(f)
                num_edges = Graph.number_of_edges()
                expected_max_weight_matching = sum(Graph.get_edge_data(*e)["weight"] for e in nx.max_weight_matching(Graph))
                start_time = time.perf_counter()
                max_weight_matching, max_weight = greedy_heuristic_mix_max_weight_matching(Graph)
                execution_time = time.perf_counter() - start_time
                results.append([
                    n_nodes,
                    edge_density,
                    num_edges,
                    subsets_counter,
                    execution_time,
                    0,
                    max_weight == expected_max_weight_matching
                ])
                
                num_subset_by_edge_density_by_nodes[edge_density][n_nodes]= subsets_counter
                num_ismatching_by_edge_density_by_nodes[edge_density][n_nodes]= is_matching_counter
                execution_time_by_edge_density_by_nodes[edge_density][n_nodes]= execution_time

                # print("\033c", end="")
                # print(tabulate(results, headers=headers, tablefmt="grid"))

    # ---------
    # Plotting |
    # ---------

    # Number of Subsets - Just one subset right? The one I go with

    # is_matching counter
    # plt.grid(True)
    # plt.xlabel('Number of Vertices')
    # plt.ylabel('is_matching')
    # for edge_density, results_by_node in num_ismatching_by_edge_density_by_nodes.items():
    #     plt.plot(results_by_node.keys(), results_by_node.values(), marker='o')
    # plt.show()

    # Execution Time
    plt.grid(True)
    plt.xlabel('Number of Vertices')
    plt.ylabel('Execution Time (s)')
    for edge_density, results_by_node in execution_time_by_edge_density_by_nodes.items():
        plt.plot(results_by_node.keys(), results_by_node.values(), marker='o')
    plt.show()
    




