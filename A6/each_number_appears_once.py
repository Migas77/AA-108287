import numpy as np


def first_repeated_index(lst):
    seen = {}
    for index, element in enumerate(lst):
        if element in seen:
            return index
        seen[element] = index  # Store the index of the element's first occurrence
    return -1

# Heads -> True
def dice_problem(n, dice_faces, dice_probabilities):
  possible_results = [i for i in range(100)]
  results = []
  for _ in range(n):
    appeared = set()
    count = 0
    while len(appeared) != len(dice_faces):
      count+=1
      appeared.add(np.random.choice(dice_faces, p=dice_probabilities))
    results.append(count)

  return {result: results.count(result) for result in possible_results}
  

if __name__ == "__main__":
  iterations = 1000000
  dice_probabilities = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
  dice_faces = [1,2,3,4,5,6]
  final_dict = dice_problem(iterations, dice_faces, dice_probabilities)
  print(final_dict)
  for k,v in final_dict.items():
    print(k, 100*v/iterations)
  print(sum(final_dict.values()))