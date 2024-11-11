from collections import defaultdict
from itertools import combinations
from tabulate import tabulate
import time
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import math


def is_matching(G, matching):
    """
    Check if the given set of edges is a matching in the graph G.

    :param G: A networkx.Graph instance.
    :param edges: Set of edges to check.
    :return: True if the given set of edges is a matching, False otherwise.
    """
    global is_matching_counter
    nodes = set()
    for edge in matching:
        # contar antes de cada if
        u,v = edge

        is_matching_counter += 1
        if u in nodes:
            return False
        
        is_matching_counter += 1
        if v in nodes:
            return False
            
        nodes.update(edge)
    return True

    


def exhaustive_search_max_weight_matching(G):
    """
    Find the maximum weighted matching using exhaustive search.

    :param G: A networkx.Graph instance with weighted edges.
    :return:
    - max_weight_matching: Set of edges in the maximum weighted matching.
    - max_weight: The total weight of the maximum weighted matching.
    """
    global subsets_counter
    max_weight = 0
    max_weight_matching = []

    # Get all edges with weights
    edges = G.edges(data="weight")
    num_edges = len(edges)
    # Generate all subsets of edges (with size 1, 2, 3, ..., len(edges))
    for r in range(1, num_edges + 1):
        subsets_counter += math.comb(num_edges, r)
        for subset in combinations(edges, r):
            # Extract edge pairs
            # no need to extract the weights as I'm using nx.is_matching
            edge_set = {(u, v) for u, v, _ in subset}

            # Check if the subset is a valid matching
            if is_matching(G, edge_set):
                # Calculate the total weight of the matching
                # and update if found maximum weight
                weight = sum(weight for _, _, weight in subset)
                if weight > max_weight:
                    max_weight = weight
                    max_weight_matching = edge_set

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
    for n_nodes in range(4, 9):
        for edge_density in [0.125, 0.25, 0.5, 0.75]:
            is_matching_counter = 0
            subsets_counter = 0
            edge_density_str = str(edge_density).replace('.', '')
            with open(f"generated_graphs/graph_{n_nodes}_{edge_density_str}.gpickle", "rb") as f:
                Graph = pickle.load(f)
                num_edges = Graph.number_of_edges()
                expected_max_weight_matching = sum(Graph.get_edge_data(*e)["weight"] for e in nx.max_weight_matching(Graph))
                start_time = time.perf_counter()
                max_weight_matching, max_weight = exhaustive_search_max_weight_matching(Graph)
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
                print(tabulate(results, headers=headers, tablefmt="grid"))

    # ---------
    # Plotting |
    # ---------

    # Number of Subsets
    # plt.grid(True)
    # plt.xlabel('Number of Vertices')
    # plt.ylabel('Number of Subsets')
    # for edge_density, results_by_node in num_subset_by_edge_density_by_nodes.items():
    #     plt.plot(results_by_node.keys(), results_by_node.values(), marker='o')
    # plt.show()

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
    
    


    # plt.figure(figsize=(10, 6))
    # plt.plot(num_edges, basic_operations, marker='o', color='b')
    # plt.ylim(0, 60000)
    # plt.legend()
    # plt.xlabel('Number of Edges')
    # plt.ylabel('Execution Time (s)')
    # plt.title('Execution Time vs Number of Edges')
    # plt.grid(True)
    # plt.show()

    

    # pos = nx.get_node_attributes(Graph, 'pos')
    # labels = nx.get_edge_attributes(Graph, 'weight')
    # nx.draw(Graph, pos, with_labels=True)
    # nx.draw_networkx_edge_labels(Graph, pos, edge_labels=labels)
    # plt.plot()
    # plt.show()

# Edmond Blossom Algorithm
# networkX max_weight_matching
"""
(1) Basic Operations:
- check if it is a matching (nx.is_matching)
- calculate the total weight of the matching
(2) Execution Time: Using time.time()
(3) Number of solutions/configurations testes
Also checked if the weight of the result is the same as the expected result (built-in function of networkX)
"""