import time
import random

random.seed(108287)

def random_max_weighted_matching_with_vertices_degree_heuristic(G, max_iter=10000, time_limit=None):
  # Sort edges by vertex degree (sum of both vertices degrees) - in ascending order
  return random_max_weighted_matching_with_heuristic(
    G, lambda x: G.degree(x[0]) + G.degree(x[1]),
    max_iter=max_iter, time_limit=time_limit
  )

def random_max_weighted_matching_with_max_weight_heuristic(G, max_iter=10000, time_limit=None):
  return random_max_weighted_matching_with_heuristic(
    G, lambda x: x[2], reverse=True,
    max_iter=max_iter, time_limit=time_limit
  )

def random_max_weighted_matching_with_mix_heuristic(G, max_iter=10000, time_limit=None):
  return random_max_weighted_matching_with_heuristic(
    G, lambda x: x[2] / (G.degree(x[0]) + G.degree(x[1]) + 1), reverse=True,
    max_iter=max_iter, time_limit=time_limit
  )

def random_max_weighted_matching_with_heuristic(G, criteria, reverse=False, max_iter=10000, time_limit=None):
  start_time = time.perf_counter()
  best_matching = set()
  max_weight = 0
  total_edges = G.number_of_edges()
  edges_per_portion = int(total_edges * 0.2)
  
  # Generate a random order of edges
  edges = list(G.edges(data="weight"))
  random.shuffle(edges)
  
  # Create the matching greedily by parsing random edges
  current_matching = set()
  matched_vertices = set()
  current_weight = 0
  
  for i in range(0, total_edges, edges_per_portion):
    max_iter -= 1
    if max_iter == 0 or (time_limit and (time.perf_counter() - start_time) > time_limit):
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
      best_matching = current_matching

  return best_matching, max_weight