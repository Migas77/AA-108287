import numpy as np
from tabulate import tabulate
import pickle
import random
import time
import networkx as nx

def random_max_weighted_matching(G, max_iterations=10000, time_limit=None):
  start_time = time.perf_counter()
  best_matching = set()
  best_weight = 0
  graph_edges = G.edges(data="weight")
  total_weight = sum(w for _, _, w in graph_edges)

  for _ in range(max_iterations):
    if time_limit and (time.perf_counter() - start_time) > time_limit:
      break

    # Generate a random order of edges
    edges = list(graph_edges)
    random.shuffle(edges)
    
    # Create a matching greedily
    current_matching = set()
    used_nodes = set()
    current_weight = 0
    remaining_weight = total_weight
    
    for edge in edges:
      u, v, w = edge
      remaining_weight -= w
      if current_weight + remaining_weight <= best_weight:
        break
      if u not in used_nodes and v not in used_nodes:
        used_nodes.add(u)
        used_nodes.add(v)
        current_matching.add(edge)
        current_weight += w

    # Update best matching if current is better
    if current_weight > best_weight:
      best_weight = current_weight
      best_matching = current_matching

  return best_matching, best_weight

def random_max_weighted_matching_with_vertices_degree_heuristic(G, max_iterations=10000, time_limit=None):
  # Sort edges by vertex degree (sum of both vertices degrees) - in ascending order
  return random_max_weighted_matching_with_heuristic(
    G, lambda x: G.degree(x[0]) + G.degree(x[1]),
    max_iterations=max_iterations, time_limit=time_limit
  )

def random_max_weighted_matching_with_max_weight_heuristic(G, max_iterations=10000, time_limit=None):
  return random_max_weighted_matching_with_heuristic(
    G, lambda x: x[2], reverse=True,
    max_iterations=max_iterations, time_limit=time_limit
  )

def random_max_weighted_matching_with_mix_heuristic(G, max_iterations=10000, time_limit=None):
  return random_max_weighted_matching_with_heuristic(
    G, lambda x: x[2] / (G.degree(x[0]) + G.degree(x[1]) + 1), reverse=True,
    max_iterations=max_iterations, time_limit=time_limit
  )

def random_max_weighted_matching_with_heuristic(G, criteria, reverse=False, max_iterations=10000, time_limit=None):
  start_time = time.perf_counter()
  max_weight_matching = set()
  max_weight = 0
  total_edges = G.number_of_edges()
  edges_per_portion = int(total_edges * 0.2)
  
  # Generate a random order of edges
  edges = list(G.edges(data="weight"))
  random.shuffle(edges)
  
  # Create a matching greedily
  current_matching = set()
  matched_vertices = set()
  current_weight = 0
  
  for i in range(0, total_edges, edges_per_portion):
    max_iterations -= 1
    if max_iterations == 0 or (time_limit and (time.perf_counter() - start_time) > time_limit):
      break

    sorted_edges_portion = sorted(edges[i:i+edges_per_portion], key=criteria, reverse=reverse)

    for edge in sorted_edges_portion:
      u, v, w = edge
      if u not in matched_vertices and v not in matched_vertices:
        matched_vertices.add(u)
        matched_vertices.add(v)
        current_matching.add(edge)
        current_weight += w

    # Update best matching if current is better
    if current_weight > max_weight:
      max_weight = current_weight
      max_weight_matching = current_matching

  return max_weight_matching, max_weight


def probabilistic_greedy_search(G, max_iterations=1000, time_limit=None):
  """
  Probabilistic Greedy Search for Maximum Weighted Matching.
  """
  start_time = time.time()
  best_matching = set()
  best_weight = 0
  edges = list(G.edges(data="weight"))
  # Compute probabilities for each edge based on weight
  total_weight = sum(w for _, _, w in edges)
  probabilities = [w / total_weight for _, _, w in edges]

  for _ in range(max_iterations):
    if time_limit and (time.time() - start_time) > time_limit:
      break

    current_matching = set()
    used_nodes = set()
    current_weight = 0
    remaining_weight = total_weight

    # Probabilistic selection of edges
    random_edge_incides = np.random.choice(len(edges), size=len(edges), p=probabilities, replace=False)

    # Greedy addition of edges from the probabilistically selected order
    for edge_indice in random_edge_incides:
      max_iterations -= 1
      if max_iterations==0 or (time_limit and (time.time() - start_time) > time_limit):
        break

      edge = edges[edge_indice]
      u, v, w = edge
      remaining_weight -= w
      if current_weight + remaining_weight <= best_weight:
        break
      if u not in used_nodes and v not in used_nodes:
        current_matching.add(edge)
        used_nodes.add(u)
        used_nodes.add(v)
        current_weight += w

    # Update best matching if current is better
    if current_weight > best_weight:
      best_weight = current_weight
      best_matching = current_matching

  return best_matching, best_weight


if __name__ == "__main__":
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
        max_weight_matching, max_weight = probabilistic_greedy_search(Graph)
        # max_weight_matching = path_growing_algorithm(Graph)
        # max_weight = sum(Graph.get_edge_data(*e)["weight"] for e in max_weight_matching)
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
        ])


      # delete output terminal
      # print("\033c", end="")
      print(tabulate(results, headers=headers, tablefmt="grid"))
      # print the number of correct results
      print(f"Correct results: {sum(1 for _, _, _, _, _, _, (a, b, c), _ in results if c)} of {len(results)}")
      print(f"Mean Precision: {np.mean([result[-2][0]/result[-2][1] for result in results])}")
