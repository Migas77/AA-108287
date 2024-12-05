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


@click.command()
@click.option('--algorithm', type=click.Choice(['random', 'random_with_heuristic', 'probabilistic_greedy']), required = True)
def run_algorithm(algorithm):
  algorithm_func = {
    'random': random_max_weighted_matching,
    'random_with_heuristic': random_max_weighted_matching_with_mix_heuristic,
    'probabilistic_greedy': probabilistic_greedy_search
  }[algorithm]
  headers = ['#Nodes', 'Edge Density', '#Edges', '#All Solutions', '#Solutions Tested', '#Basic operations', 'Execution Time (s)', 'isExpectedResult', 'Precision']
  results = []
  node_range = range(4, 1000, 25)

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
        max_weight_matching, max_weight = algorithm_func(Graph)
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

      # print(tabulate(results, headers=headers, tablefmt="grid"))

      if algorithm == 'random' and n_nodes == 154 and edge_density == 0.75:
        # 154, 0.75

        # Print results
        print(tabulate(results, headers=headers, tablefmt="grid"))
        print(f"Correct results: {sum(1 for _, _, _, _, _, _, _, (a, b, c), _ in results if c)} of {len(results)}")
        print(f"Mean Precision: {np.mean([result[-2][0]/result[-2][1] for result in results])}")

        node_colors = { 4: 'tab:blue', 29: 'tab:orange', 54: 'tab:green', 79: 'tab:red', 104: 'tab:purple', 129: 'tab:brown', 154: 'tab:pink'}

        # Plot number of solutions tested
        number_edges = [result[2] for result in results]
        number_filtered_solutions_tested = [result[4] for result in results]
        plt.figure(figsize=(10, 6))
        for i in range(0, len(number_edges), 4):
          batch_edges = number_edges[i:i+4]
          batch_solutions_tested = number_filtered_solutions_tested[i:i+4]
          plt.scatter(batch_edges, batch_solutions_tested, color=node_colors[node_range[i//4]], label=f'n_nodes = {node_range[i//4]}')
        plt.xlabel('Number of Edges')
        plt.ylabel('Number of Solutions Tested')
        plt.title('Number of Solutions Tested by Number of Edges')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

        # Plot number of basic operations by number of edges with linear fit
        number_basic_operations = [result[5] for result in results]
        plt.figure(figsize=(10, 6))
        for i in range(0, len(number_edges), 4):
          batch_edges = number_edges[i:i+4]
          batch_basic_operations = number_basic_operations[i:i+4]
          plt.scatter(batch_edges, batch_basic_operations, color=node_colors[node_range[i//4]], label=f'n_nodes = {node_range[i//4]}', zorder=2)
        all_edges = np.array(number_edges).reshape(-1, 1)
        all_basic_operations = np.array(number_basic_operations)
        model = LinearRegression()
        model.fit(all_edges, all_basic_operations)
        predicted_operations = model.predict(all_edges)
        plt.plot(number_edges, predicted_operations, color='k', label=f'Linear Fit: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}', zorder=1)
        plt.xlabel('Number of Edges')
        plt.ylabel('Number of Basic Operations')
        plt.title('Number of Basic Operations by Number of Edges')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()

        # Plot execution time by number of edges with linear fit
        execution_times = [result[6] for result in results]
        plt.figure(figsize=(10, 6))
        for i in range(0, len(number_edges), 4):
          batch_edges = number_edges[i:i+4]
          batch_execution_times = execution_times[i:i+4]
          plt.scatter(batch_edges, batch_execution_times, color=node_colors[node_range[i//4]], label=f'n_nodes = {node_range[i//4]}', zorder=2)
        all_edges = np.array(number_edges).reshape(-1, 1)
        all_execution_times = np.array(execution_times)
        model = LinearRegression()
        model.fit(all_edges, all_execution_times)
        predicted_times = model.predict(all_edges)
        plt.plot(number_edges, predicted_times, color='k', label=f'Linear Fit: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}', zorder=1)
        plt.xlabel('Number of Edges')
        plt.ylabel('Execution Time (s)')
        plt.title('Execution Time by Number of Edges')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()

        # Plot precision by number of edges
        precision = [result[-1] for result in results]
        plt.figure(figsize=(10, 6))
        for i in range(0, len(number_edges), 4):
          batch_edges = number_edges[i:i+4]
          batch_precision = precision[i:i+4]
          plt.scatter(batch_edges, batch_precision, color=node_colors[node_range[i//4]], label=f'n_nodes = {node_range[i//4]}')
        plt.xlabel('Number of Edges')
        plt.ylabel('Precision')
        plt.title('Precision by Number of Edges')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

        return

        
  if algorithm == 'probabilistic_greedy':
    # Print results
    print(tabulate(results, headers=headers, tablefmt="grid"))
    print(f"Correct results: {sum(1 for _, _, _, _, _, _, _, (a, b, c), _ in results if c)} of {len(results)}")
    print(f"Mean Precision: {np.mean([result[-2][0]/result[-2][1] for result in results])}")

    # Generate random colors for the node range
    node_colors = {4: 'red', 29: 'blue', 54: 'cyan', 79: 'green', 104: 'yellow', 129: 'purple', 154: 'orange', 179: 'pink', 204: 'brown', 229: 'lightblue', 254: 'lightgreen', 279: 'darkblue', 304: 'darkgreen', 329: 'darkred', 354: 'violet', 379: 'lightgray', 404: 'darkorange', 429: 'crimson', 454: 'indigo', 479: 'gold', 504: 'teal', 529: 'lime', 554: '#D2691E', 579: 'maroon', 604: 'fuchsia', 629: 'turquoise', 654: 'lightpink', 679: 'yellowgreen', 704: 'orchid', 729: 'lavender', 754: 'slateblue', 779: 'seagreen', 804: 'chartreuse', 829: 'tomato', 854: '#8A2BE2', 879: 'mediumvioletred', 904: 'mediumpurple', 929: 'darkslategray', 954: 'darkkhaki', 979: 'lightseagreen'}

    # Plot number of solutions tested
    number_edges = [result[2] for result in results]
    number_filtered_solutions_tested = [result[4] for result in results]
    plt.figure(figsize=(10, 6))
    for i in range(0, len(number_edges), 4):
      batch_edges = number_edges[i:i+4]
      batch_solutions_tested = number_filtered_solutions_tested[i:i+4]
      plt.scatter(batch_edges, batch_solutions_tested, color=node_colors[node_range[i//4]], label=f'n_nodes = {node_range[i//4]}')
    plt.xlabel('Number of Edges')
    plt.ylabel('Number of Solutions Tested')
    plt.title('Number of Solutions Tested by Number of Edges')
    plt.legend(loc='upper right', ncol=3)
    plt.grid(True)
    plt.show()

    # Plot number of basic operations by number of edges
    number_basic_operations = [result[5] for result in results]
    plt.figure(figsize=(10, 6))
    for i in range(0, len(number_edges), 4):
      batch_edges = number_edges[i:i+4]
      batch_basic_operations = number_basic_operations[i:i+4]
      plt.scatter(batch_edges, batch_basic_operations, color=node_colors[node_range[i//4]], label=f'n_nodes = {node_range[i//4]}')
    plt.xlabel('Number of Edges')
    plt.ylabel('Number of Basic Operations')
    plt.title('Number of Basic Operations by Number of Edges')
    plt.legend(loc='upper right', ncol=3)
    plt.grid(True)
    plt.show()

    # Plot execution time by number of edges
    execution_times = [result[6] for result in results]
    plt.figure(figsize=(10, 6))
    for i in range(0, len(number_edges), 4):
      batch_edges = number_edges[i:i+4]
      batch_execution_times = execution_times[i:i+4]
      plt.scatter(batch_edges, batch_execution_times, color=node_colors[node_range[i//4]], label=f'n_nodes = {node_range[i//4]}')
    plt.xlabel('Number of Edges')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time by Number of Edges')
    plt.legend(loc='lower right', ncol=3)
    plt.grid(True)
    plt.show()

    # Plot precision by number of edges
    precision = [result[-1] for result in results]
    plt.figure(figsize=(10, 6))
    for i in range(0, len(number_edges), 4):
      batch_edges = number_edges[i:i+4]
      batch_precision = precision[i:i+4]
      plt.scatter(batch_edges, batch_precision, color=node_colors[node_range[i//4]], label=f'n_nodes = {node_range[i//4]}')
    plt.xlabel('Number of Edges')
    plt.ylabel('Precision')
    plt.title('Precision by Number of Edges')
    plt.legend(loc='lower right', ncol=3)
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
  run_algorithm()
