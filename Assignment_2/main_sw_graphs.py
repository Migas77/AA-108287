import exp
import click
import numpy as np
from tabulate import tabulate
import pickle
import time
import networkx as nx
from random_max_weighted_matching import random_max_weighted_matching
from random_max_weighted_matching_with_heuristic import random_max_weighted_matching_with_mix_heuristic
from probabilistic_greedy_search import probabilistic_greedy_search
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

import networkx as nx

def read_graph(file_path, algorithm):
    assert algorithm in ['random', 'random_with_heuristic', 'probabilistic_greedy']
    assert 'SWtinyEWD' in file_path or 'SWmediumEWD' in file_path or 'SW1000EWD' in file_path or 'SW10000EWD' in file_path

    with open(file_path, 'r') as file:
      lines = file.readlines()

    is_directed = int(lines[0].strip())  # Doesn't matter because the graph will always be made undirected
    has_weights = int(lines[1].strip())
    num_vertices = int(lines[2].strip())
    num_edges = int(lines[3].strip())
    assert has_weights == 1 # it has to have weights for the max weighted matching problem

    # Create an undirected graph
    graph = nx.Graph()

    # Set to track added edges (order doesn't matter)
    added_edges = set()

    # Read edges and add them to the graph
    for line in lines[4:]:
      parts = line.split()
      u, v = int(parts[0]), int(parts[1])

      # Skip self-loops
      if u == v:
        continue

      # Create a sorted tuple for the edge (undirected)
      edge = tuple(sorted([u, v]))

      # Check if the edge has already been added
      if edge not in added_edges:
        # Add the edge to the graph
        added_edges.add(edge)

        # Add weighted or unweighted edges
        if has_weights:
          # i asserted it above but it remains here for future reference
          weight = float(parts[2])
          
          if algorithm == 'probabilistic_greedy':
            # probabilistic_greedy has probabilities calculated based on the weights; P= wi**100 / sum(wj**100 for j in edges)
            # due to the exponent weight has to be bigger than 1 to avoid uninteded behaviour
            if 'SWtinyEWD' in file_path:
              weight = weight * 10
            elif 'SWmediumEWD' in file_path:
              weight = weight * 1000
            elif 'SW1000EWD' in file_path:
              weight = weight * 10000
            elif 'SW10000EWD' in file_path:
              weight = weight * 20000
            assert weight > 1

          graph.add_edge(u, v, weight=weight)
        else:
          graph.add_edge(u, v)

    return graph


@click.command()
@click.option('--algorithm', type=click.Choice(['random', 'random_with_heuristic', 'probabilistic_greedy']), required = True)
def run_algorithm(algorithm):
  algorithm_func = {
    'random': random_max_weighted_matching,
    'random_with_heuristic': random_max_weighted_matching_with_mix_heuristic,
    'probabilistic_greedy': probabilistic_greedy_search
  }[algorithm]
  headers = ['#Nodes', '#Edges', '#All Solutions', '#Solutions Tested', '#Basic operations', 'Execution Time (s)', 'isExpectedResult', 'Precision']
  results = []

  for filename in ["SWtinyEWD.txt", "SWmediumEWD.txt", "SW1000EWD.txt", "SW10000EWD.txt",]:
    file_path = os.path.join(os.getcwd(), 'sw_graphs', filename)
    Graph = read_graph(file_path, algorithm)
    print(f"File {file_path} read")

    exp.all_solutions_counter = 0
    exp.filtered_solutions_counter = 0
    exp.basic_operations_counter = 0

    start_time = time.perf_counter()
    max_weight_matching, max_weight = algorithm_func(Graph)
    execution_time = time.perf_counter() - start_time

    expected_max_weight = sum(Graph.get_edge_data(*e)["weight"] for e in nx.max_weight_matching(Graph))

    results.append([
      Graph.number_of_nodes(),
      Graph.number_of_edges(),
      exp.all_solutions_counter,
      exp.filtered_solutions_counter,
      exp.basic_operations_counter,
      execution_time,
      (max_weight, expected_max_weight, max_weight == expected_max_weight),
      max_weight / expected_max_weight if expected_max_weight != 0 else -1
    ])

  # Print results
  print(tabulate(results, headers=headers, tablefmt="grid"))

    
if __name__ == "__main__":
  run_algorithm()
