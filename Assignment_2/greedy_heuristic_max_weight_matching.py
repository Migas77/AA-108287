from itertools import combinations
from collections import defaultdict
import os
import numpy as np
from tabulate import tabulate
import pickle
import time
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def greedy_heuristic_vertices_degrees_max_weight_matching(G):
    # Sort edges by vertex degree (for each of the vertex of the the edge) - in ascending order
    return greedy_heuristic_max_weight_matching(G, lambda x: G.degree(x[0]) + G.degree(x[1]))

def greedy_heuristic_edges_weight_max_weight_matching(G):
    # Sort edges by weight in descending order
    return greedy_heuristic_max_weight_matching(G, lambda x: x[2], reverse=True)

def greedy_heuristic_mix_max_weight_matching(G):
    # Sort edges by weight in descending order, but prefer edges with vertices with lower degree
    return greedy_heuristic_max_weight_matching(G, lambda x: x[2] / (G.degree(x[0]) + G.degree(x[1]) + 1), reverse=True)

    
def greedy_heuristic_max_weight_matching(G, criteria, reverse=False):
    global configurations_counter, operations_counter
    max_weight_matching = []
    max_weight = 0
    matched_vertices = set()
    sorted_edges = sorted(G.edges(data="weight"), key=criteria, reverse=reverse)

    for edge in sorted_edges:
        configurations_counter += 1

        u, v, weight = edge

        operations_counter += 1
        if u not in matched_vertices:
            operations_counter += 1
            if v not in matched_vertices:
                matched_vertices.add(u)
                matched_vertices.add(v)
                max_weight_matching.append(edge)
                max_weight += weight
            
    return max_weight_matching, max_weight


if __name__ == '__main__':
    if not os.path.exists('charts/greedy'):
        os.makedirs('charts/greedy')
    configurations_counter = 0
    operations_counter = 0
    nodes_range = range(20, 1000, 25)
    node_colors = {node: cm.rainbow(i / len(nodes_range)) for i, node in enumerate(nodes_range)}
    headers = ['#Nodes', 'Edge Density', '#Edges', '#Configs Tested', '#Basic Operations', 'Execution Time (s)', 'expected', 'actual', 'Precision']
    results = [] 
    execution_time_by_edge_density_by_nodes = defaultdict(defaultdict)
    for n_nodes in nodes_range:
        for edge_density in [0.125, 0.25, 0.5, 0.75]:
            configurations_counter = 0
            operations_counter = 0
            edge_density_str = str(edge_density).replace('.', '')
            with open(f"generated_graphs/graph_{n_nodes}_{edge_density_str}.gpickle", "rb") as f:
                Graph = pickle.load(f)
                num_edges = Graph.number_of_edges()
                if num_edges <= 0:
                    continue
                expected_max_weight_matching = sum(Graph.get_edge_data(*e)["weight"] for e in nx.max_weight_matching(Graph))
                start_time = time.perf_counter()
                max_weight_matching, max_weight = greedy_heuristic_mix_max_weight_matching(Graph)
                execution_time = time.perf_counter() - start_time
                results.append([
                    n_nodes,
                    edge_density,
                    num_edges,
                    configurations_counter,
                    operations_counter,
                    execution_time,
                    expected_max_weight_matching,
                    max_weight,
                    # greedy_result / optimal_result
                    max_weight / expected_max_weight_matching if expected_max_weight_matching > 0 else -1
                ])
                execution_time_by_edge_density_by_nodes[edge_density][n_nodes]= execution_time
                print(tabulate(results, headers=headers, tablefmt="grid"))
                print(f"Mean Precision: {np.mean([result[-1] for result in results])}")

    # ----------------
    # Tabular Results |
    # ----------------
    

    # ---------
    # Plotting |
    # ---------

    # Number of Configurations
    handle_by_label = {}
    for result in results:
        number_nodes = result[0]
        number_edges = result[2]
        number_configurations = result[3]
        line, = plt.plot(number_edges, number_configurations, marker='o', label=f'n_nodes={number_nodes}', color=node_colors[number_nodes])
        label = f'n_nodes={number_nodes}'
        if label not in handle_by_label:
            handle_by_label[label] = line

    handle_by_label = dict(sorted(handle_by_label.items(), key=lambda x: int(x[0].split('=')[-1])))
    plt.legend(labels=handle_by_label.keys(), handles=handle_by_label.values())
    # NO LOG SCALE
    plt.grid(True)
    plt.xlabel('Number of Edges')
    plt.ylabel('Number of Configurations Tested')
    plt.title('Greedy: Number of Configurations Tested by Number of Edges')
    plt.gca().title.set_size(9)
    plt.show()

    # Operations Call Count
    plt.clf()
    handle_by_label = {}
    for result in results:
        number_nodes = result[0]
        number_edges = result[2]
        number_operations = result[4]
        line, = plt.plot(number_edges, number_operations, marker='o', label=f'n_nodes={number_nodes}', color=node_colors[number_nodes])
        label = f'n_nodes={number_nodes}'
        if label not in handle_by_label:
            handle_by_label[label] = line

    handle_by_label = dict(sorted(handle_by_label.items(), key=lambda x: int(x[0].split('=')[-1])))
    plt.legend(labels=handle_by_label.keys(), handles=handle_by_label.values())
    # NO LOG SCALE
    plt.grid(True)
    plt.xlabel('Number of Edges')
    plt.ylabel('Number of Basic Operations')
    plt.title('Greedy: Number of Basic Operations by Number of Edges')
    plt.gca().title.set_size(9)
    plt.show()

    # Execution Time
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

    handle_by_label = dict(sorted(handle_by_label.items(), key=lambda x: int(x[0].split('=')[-1])))
    plt.legend(labels=handle_by_label.keys(), handles=handle_by_label.values())
    # NO LOG SCALE
    plt.grid(True)
    plt.xlabel('Number of Edges')
    plt.ylabel('Execution Time (s)')
    plt.title('Greedy: Execution Time by Number of Edges')
    plt.gca().title.set_size(9)
    plt.show()

    # By edge density
    edge_density_colors = {0.125: 'r', 0.25: 'g', 0.5: 'b', 0.75: 'y'}
    # With log scale
    plt.grid(True)
    plt.xlabel('Number of Vertices')
    plt.ylabel('Execution Time (s) (log scale)')
    for edge_density, results_by_node in execution_time_by_edge_density_by_nodes.items():
        plt.plot(results_by_node.keys(), results_by_node.values(), marker='o', label=f'Edge Density={edge_density}', color=edge_density_colors[edge_density])
        plt.yscale('log')
    plt.legend()
    plt.show()
    




