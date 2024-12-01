import click
import numpy as np
from tabulate import tabulate
import pickle
import time
import networkx as nx
from random_max_weighted_matching import random_max_weighted_matching
from random_max_weighted_matching_with_heuristic import random_max_weighted_matching_with_mix_heuristic
from probabilistic_greedy_search import probabilistic_greedy_search

@click.command()
@click.option('--algorithm', type=click.Choice(['random', 'random_with_heuristic', 'probabilistic_greedy']), required = True)
def run_algorithm(algorithm):
  algorithm_func = {
    'random': random_max_weighted_matching,
    'random_with_heuristic': random_max_weighted_matching_with_mix_heuristic,
    'probabilistic_greedy': probabilistic_greedy_search
  }[algorithm]
  headers = ['#Nodes', 'Edge Density', '#Edges', '#Subsets Tested', '#is_matching operations', 'Execution Time (s)', 'isExpectedResult', 'Precision']
  results = []

  for n_nodes in range(20, 1000, 25):
    for edge_density in [0.125, 0.25, 0.5, 0.75]:
      # Load Graph
      edge_density_str = str(edge_density).replace('.', '')
      with open(f"generated_graphs/graph_{n_nodes}_{edge_density_str}.gpickle", "rb") as f:
        Graph = pickle.load(f)
        num_edges = Graph.number_of_edges()
        if num_edges <= 0:
            continue
        
        start_time = time.perf_counter()
        max_weight_matching, max_weight = algorithm_func(Graph)
        execution_time = time.perf_counter() - start_time

        expected_max_weight = sum(Graph.get_edge_data(*e)["weight"] for e in nx.max_weight_matching(Graph))

        results.append([
          n_nodes,
          edge_density,
          num_edges,
          0,
          0,
          execution_time,
          (max_weight, expected_max_weight, max_weight == expected_max_weight),
          max_weight / expected_max_weight if expected_max_weight != 0 else -1
          # num_entered
        ])


      # delete output terminal
      # print("\033c", end="")
      print(tabulate(results, headers=headers, tablefmt="grid"))
      # print the number of correct results
      print(f"Correct results: {sum(1 for _, _, _, _, _, _, (a, b, c), _ in results if c)} of {len(results)}")
      print(f"Mean Precision: {np.mean([result[-2][0]/result[-2][1] for result in results])}")


if __name__ == "__main__":
  run_algorithm()
