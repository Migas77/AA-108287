import time
import random

def random_max_weighted_matching(G, max_iter=10000, time_limit=None):
  start_time = time.perf_counter()
  best_matching = set()
  edges = list(G.edges(data="weight"))
  total_weight = sum(w for _, _, w in edges)
  max_weight = 0

  for i in range(max_iter):
    if time_limit and (time.perf_counter() - start_time) > time_limit:
      break

    # Generate a random order of edges
    random.shuffle(edges)
    
    # Create the matching greedily by parsing random edges
    current_matching = set()
    matched_vertices = set()
    current_weight = 0
    remaining_weight = total_weight
    
    for edge in edges:
      u, v, w = edge
      remaining_weight -= w
      if current_weight + remaining_weight <= max_weight:
        break
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


