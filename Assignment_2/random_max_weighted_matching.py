import time
import random
import exp

random.seed(108287)

def random_max_weighted_matching(G, max_iter=10000, time_limit=None):
  start_time = time.perf_counter()
  best_matching = set()
  shuffled_edges = set()
  edges = list(G.edges(data="weight"))
  total_weight = sum(w for _, _, w in edges)
  max_weight = 0

  for i in range(max_iter):
    if time_limit and (time.perf_counter() - start_time) > time_limit:
      break

    exp.all_solutions_counter += 1

    # Generate a random order of edges
    random.shuffle(edges)
    tuple_edges = tuple(edges)
    if (tuple_edges in shuffled_edges):
      continue
    shuffled_edges.add(tuple_edges)

    exp.filtered_solutions_counter += 1
    
    # Create the matching greedily by parsing random edges
    current_matching = set()
    matched_vertices = set()
    current_weight = 0
    remaining_weight = total_weight
    for edge in edges:
      u, v, w = edge
      remaining_weight -= w
      if current_weight + remaining_weight < max_weight:
        break
      
      exp.basic_operations_counter += 1
      if u not in matched_vertices:
        exp.basic_operations_counter += 1
        if v not in matched_vertices:
          matched_vertices.add(u)
          matched_vertices.add(v)
          current_matching.add(edge)
          current_weight += w

    # Update best matching if current is better
    if current_weight > max_weight:
      max_weight = current_weight
      best_matching = current_matching

  return best_matching, max_weight


"""
Respostas às dúvidas:
Usar uma constante para a complexidade do np.random se não se sabe
Usar 2 counters para os subsets
O resto está bem
"""