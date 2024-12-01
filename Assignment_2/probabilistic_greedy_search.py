import time
import random
import numpy as np

def probabilistic_greedy_search(G, max_iter=1000, time_limit=2):
  """
  Probabilistic Greedy Search for Maximum Weighted Matching.
  """
  start_time = time.time()
  best_matching = set()
  best_weight = 0
  edges = list(G.edges(data="weight"))

  total_weight = sum(w for _, _, w in edges)
  probabilities = [w / total_weight for u, v, w in edges]
  # Adjust probabilities to give higher weight edges a higher probability
  adjusted_weights = [w ** 10 for _, _, w in edges]
  total_adjusted_weight = sum(adjusted_weights)
  probabilities = [w / total_adjusted_weight for w in adjusted_weights]
  
  for i in range(max_iter):
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
      max_iter -= 1
      if max_iter==0 or (time_limit and (time.time() - start_time) > time_limit):
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