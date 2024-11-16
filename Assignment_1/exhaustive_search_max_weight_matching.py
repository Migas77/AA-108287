from collections import defaultdict
from itertools import combinations
import os
import numpy as np
from tabulate import tabulate
import time
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit


def is_matching(matching):
    """
    Check if the given set of edges is a matching in the graph G.

    :param G: A networkx.Graph instance.
    :param edges: Set of edges to check.
    :return: True if the given set of edges is a matching, False otherwise.
    """
    global is_matching_counter
    nodes = set()
    for edge in matching:
        # Extract edge pair
        # no need to extract the weights
        u,v, weight = edge

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
    max_weight_matching = set()
    max_weight = 0

    # Get all edges with weights
    edges = G.edges(data="weight")
    num_edges = len(edges)
    # Generate all subsets of edges (with size 1, 2, 3, ..., len(edges))
    for r in range(1, num_edges + 1):
        for edge_subset in combinations(edges, r):
            subsets_counter += 1

            # Check if the subset is a valid matching
            if is_matching(edge_subset):
                # Calculate the total weight of the matching
                # and update if found maximum weight
                weight = sum(weight for _, _, weight in edge_subset)
                if weight > max_weight:
                    max_weight_matching = edge_subset
                    max_weight = weight

    return max_weight_matching, max_weight

def model(m, a, b):
    return a + b * m * (2 ** m)

def model2(m, a, b, c): 
    return a * m * (2 ** (b * m)) + c

def model3(x, a, b, c):
    return a + b * np.exp(c * x)

if __name__ == '__main__':
    if not os.path.exists('charts'):
        os.makedirs('charts')
    subsets_counter = 0
    is_matching_counter = 0
    headers = ['#Nodes', 'Edge Density', '#Edges', '#Subsets Tested', '#is_matching operations', 'Execution Time (s)', 'isExpectedResult']
    results = [] 
    node_colors = { 4: 'r', 5: 'g', 6: 'b', 7: 'c', 8: 'm', 9: 'y',}
    num_subset_by_num_edges = defaultdict(defaultdict)
    num_subset_by_edge_density_by_nodes = defaultdict(defaultdict)
    num_ismatching_by_edge_density_by_nodes = defaultdict(defaultdict)
    execution_time_by_edge_density_by_nodes = defaultdict(defaultdict)
    """
    num_vertices_4: {
        densidade_0.125: [resultado1, resultado2, ...]
    }
    """
    for n_nodes in range(4, 10):
        for edge_density in [0.125, 0.25, 0.5, 0.75]:
            is_matching_counter = 0
            subsets_counter = 0
            edge_density_str = str(edge_density).replace('.', '')
            with open(f"generated_graphs/graph_{n_nodes}_{edge_density_str}.gpickle", "rb") as f:
                Graph = pickle.load(f)
                num_edges = Graph.number_of_edges()
                if num_edges <= 0:
                    continue
                expected_max_weight_matching = sum(Graph.get_edge_data(*e)["weight"] for e in nx.max_weight_matching(Graph))
                start_time = time.perf_counter()
                max_weight_matching, max_weight = exhaustive_search_max_weight_matching(Graph)
                execution_time = time.perf_counter() - start_time
                results.append([
                    n_nodes,
                    edge_density,
                    num_edges,
                    subsets_counter,
                    is_matching_counter,
                    execution_time,
                    max_weight == expected_max_weight_matching
                ])
                num_subset_by_num_edges[num_edges] = {
                    "n_nodes": n_nodes,
                    "subsets_counter": subsets_counter
                }
                num_subset_by_edge_density_by_nodes[edge_density][n_nodes]= subsets_counter
                num_ismatching_by_edge_density_by_nodes[edge_density][n_nodes]= is_matching_counter
                execution_time_by_edge_density_by_nodes[edge_density][n_nodes]= execution_time

    print(tabulate(results, headers=headers, tablefmt="grid"))
    execution_time_by_num_edges = []

    # ---------
    # Plotting |
    # ---------

    # Number of Subsets
    num_subset_by_num_edges = dict(sorted(num_subset_by_num_edges.items()))
    plt.plot(
        num_subset_by_num_edges.keys(),
        [entry["subsets_counter"] for entry in num_subset_by_num_edges.values()],
        color='k',
    )

    handle_by_label = {}
    for number_edges, entry in num_subset_by_num_edges.items():
        number_nodes = entry["n_nodes"]
        label = f'n_nodes={number_nodes}'
        line, = plt.plot(
            number_edges, entry["subsets_counter"], marker='o', label=label,
            color=node_colors[number_nodes]
        )
        if label not in handle_by_label:
            handle_by_label[label] = line

    handle_by_label = dict(sorted(handle_by_label.items()))
    plt.legend(labels=handle_by_label.keys(), handles=handle_by_label.values())
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel('Number of Edges')
    plt.ylabel('Number of Subsets (log scale)')
    plt.title('Exhaustive Search: Number of Subsets tested by Number of Edges')
    plt.gca().title.set_size(10)
    plt.show()
    # plt.savefig('charts/num_subsets_by_num_edges.png')

    # is_matching operations count
    plt.clf()
    handle_by_label = {}
    for result in results:
        number_nodes = result[0]
        number_edges = result[2]
        number_matching_operations = result[4]
        line, = plt.plot(number_edges, number_matching_operations, marker='o', label=f'n_nodes={number_nodes}', color=node_colors[number_nodes])
        label = f'n_nodes={number_nodes}'
        if label not in handle_by_label:
            handle_by_label[label] = line
    
    handle_by_label = dict(sorted(handle_by_label.items()))
    plt.legend(labels=handle_by_label.keys(), handles=handle_by_label.values())
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel('Number of Edges')
    plt.ylabel('is_matching() operations count (log scale)')
    plt.title('Exhaustive Search: is_matching() operations count by Number of Edges')
    plt.gca().title.set_size(10)
    # plt.show()
    plt.savefig('charts/is_matching_counter_by_num_edges.png')

    plt.clf()
    handle_by_label = {}
    for result in results:
        number_nodes = result[0]
        number_edges = result[2]
        execution_time = result[5]
        line, = plt.plot(number_edges, execution_time, marker='o', label=f'n_nodes={number_nodes}', color=node_colors[number_nodes])
        label = f'n_nodes={number_nodes}'
        if label not in handle_by_label:
            handle_by_label[label] = line

    handle_by_label = dict(sorted(handle_by_label.items()))
    plt.legend(labels=handle_by_label.keys(), handles=handle_by_label.values())
    plt.yscale('log')
    plt.grid(True)
    plt.xlabel('Number of Edges')
    plt.ylabel('Execution Time (s) (log scale)')
    plt.title('Exhaustive Search: Execution Time by Number of Edges')
    plt.show()

    # pos = nx.get_node_attributes(Graph, 'pos')
    # labels = nx.get_edge_attributes(Graph, 'weight')
    # nx.draw(Graph, pos, with_labels=True)
    # nx.draw_networkx_edge_labels(Graph, pos, edge_labels=labels)
    # plt.plot()
    # plt.show()
