import time
import random
import numpy as np
import exp

random.seed(108287)

def probabilistic_greedy_search(G, exponent=100, max_iter=1000, time_limit=2):
  """
  Probabilistic Greedy Search for Maximum Weighted Matching.
  """
  start_time = time.time()
  best_matching = set()
  edges = list(G.edges(data="weight"))
  num_edges = len(edges)
  random_indices = set()
  total_weight = sum(w for _, _, w in edges)
  max_weight = 0
  
  # Adjust probabilities to give higher weight edges a higher probability
  adjusted_weights = [w ** exponent for _, _, w in edges]
  total_adjusted_weight = sum(adjusted_weights)
  probabilities = [w / total_adjusted_weight for w in adjusted_weights]
  
  for i in range(max_iter):
    if time_limit and (time.time() - start_time) > time_limit:
      break

    exp.all_solutions_counter += 1

    # Probabilistic selection of edges
    random_edge_incides = np.random.choice(num_edges, size=num_edges, p=probabilities, replace=False)
    tuple_random_edge_indices = tuple(random_edge_incides)
    if tuple_random_edge_indices in random_indices:
      continue
    random_indices.add(tuple_random_edge_indices)

    exp.filtered_solutions_counter += 1

    # Greedy addition of edges from the probabilistically selected order
    current_matching = set()
    matched_vertices = set()
    current_weight = 0
    remaining_weight = total_weight
    for edge_indice in random_edge_incides:

      edge = edges[edge_indice]
      u, v, w = edge
      remaining_weight -= w
      if current_weight + remaining_weight < max_weight:
        break
      exp.basic_operations_counter += 1
      if u not in matched_vertices:
        exp.basic_operations_counter += 1
        if v not in matched_vertices:
          current_matching.add(edge)
          matched_vertices.add(u)
          matched_vertices.add(v)
          current_weight += w

    # Update best matching if current is better
    if current_weight > max_weight:
      max_weight = current_weight
      best_matching = current_matching

  return best_matching, max_weight