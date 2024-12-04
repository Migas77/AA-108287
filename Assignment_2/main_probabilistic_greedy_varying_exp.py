import exp
import numpy as np
from tabulate import tabulate
import pickle
import time
import networkx as nx
from random_max_weighted_matching import random_max_weighted_matching
from random_max_weighted_matching_with_heuristic import random_max_weighted_matching_with_mix_heuristic
from probabilistic_greedy_search import probabilistic_greedy_search
import matplotlib.pyplot as plt


def run_algorithm():
  headers = ['#Nodes', 'Edge Density', '#Edges', '#All Solutions', '#Solutions Tested', '#Basic operations', 'Execution Time (s)', 'isExpectedResult', 'Precision']
  results_by_exponent = {}
  node_range = range(4, 1000, 25)

  for exponent in [1, 10, 100]:
    results = []

    for n_nodes in node_range:
      for edge_density in [0.125, 0.25, 0.5, 0.75]:
        # Load Graph
        edge_density_str = str(edge_density).replace('.', '')
        with open(f"generated_graphs/graph_{n_nodes}_{edge_density_str}.gpickle", "rb") as f:
          Graph = pickle.load(f)
          num_edges = Graph.number_of_edges()
          if num_edges <= 0:
              continue
          
          exp.all_solutions_counter = 0
          exp.filtered_solutions_counter = 0
          exp.basic_operations_counter = 0

          start_time = time.perf_counter()
          max_weight_matching, max_weight = probabilistic_greedy_search(Graph, exponent=exponent)
          execution_time = time.perf_counter() - start_time

          expected_max_weight = sum(Graph.get_edge_data(*e)["weight"] for e in nx.max_weight_matching(Graph))

          results.append([
            n_nodes,
            edge_density,
            num_edges,
            exp.all_solutions_counter,
            exp.filtered_solutions_counter,
            exp.basic_operations_counter,
            execution_time,
            (max_weight, expected_max_weight, max_weight == expected_max_weight),
            max_weight / expected_max_weight if expected_max_weight != 0 else -1
          ])

    results_by_exponent[exponent] = results
    # Print results
    print(f"Results for exponent = {exponent}")
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print(f"Correct results: {sum(1 for _, _, _, _, _, _, _, (a, b, c), _ in results if c)} of {len(results)}")
    print(f"Mean Precision: {np.mean([result[-2][0]/result[-2][1] for result in results])}")

  # Plot number of solutions tested
  plt.figure(figsize=(10, 6))
  for index, (exponent, exp_results) in enumerate(results_by_exponent.items()):
    number_edges = [result[2] for result in exp_results]
    number_filtered_solutions_tested = [result[4] for result in exp_results]
    plt.scatter(number_edges, number_filtered_solutions_tested, color=['red', 'green', 'blue'][index], label=f'Exponent = {exponent}')
  plt.xlabel('Number of Edges')
  plt.ylabel('Number of Solutions Tested')
  plt.title('Number of Solutions Tested by Number of Edges')
  plt.legend(loc='upper right')
  plt.grid(True)
  plt.show()

  # Plot number of basic operations by number of edges
  plt.figure(figsize=(10, 6))
  for index, (exponent, exp_results) in enumerate(results_by_exponent.items()):
    number_edges = [result[2] for result in exp_results]
    number_basic_operations = [result[5] for result in exp_results]
    plt.scatter(number_edges, number_basic_operations, color=['red', 'green', 'blue'][index], label=f'Exponent = {exponent}')
  plt.xlabel('Number of Edges')
  plt.ylabel('Number of Basic Operations')
  plt.title('Number of Basic Operations by Number of Edges')
  plt.legend(loc='upper right')
  plt.grid(True)
  plt.show()

  # Plot execution time by number of edges
  plt.figure(figsize=(10, 6))
  for index, (exponent, exp_results) in enumerate(results_by_exponent.items()):
    number_edges = [result[2] for result in exp_results]
    execution_times = [result[6] for result in exp_results]
    plt.scatter(number_edges, execution_times, color=['red', 'green', 'blue'][index], label=f'Exponent = {exponent}')
  plt.xlabel('Number of Edges')
  plt.ylabel('Execution Time (s)')
  plt.title('Execution Time by Number of Edges')
  plt.legend(loc='lower right')
  plt.grid(True)
  plt.show()

  # Plot precision by number of edges
  plt.figure(figsize=(10, 6))
  for index, (exponent, exp_results) in enumerate(results_by_exponent.items()):
    number_edges = [result[2] for result in exp_results]
    precision = [result[-1] for result in exp_results]
    plt.scatter(number_edges, precision, color=['red', 'green', 'blue'][index], label=f'Exponent = {exponent}')
  plt.xlabel('Number of Edges')
  plt.ylabel('Precision')
  plt.title('Precision by Number of Edges')
  plt.legend(loc='upper right')
  plt.grid(True)
  plt.show()



if __name__ == "__main__":
  run_algorithm()
