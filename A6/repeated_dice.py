import numpy as np


def first_repeated_index(lst):
    seen = {}
    for index, element in enumerate(lst):
        if element in seen:
            return index
        seen[element] = index  # Store the index of the element's first occurrence
    return -1

# Heads -> True
def dice_problem(n):
  possible_results = [i for i in range(9)]
  normal_dice = [1/6,1/6,1/6,1/6,1/6,1/6,]
  dice = [1,2,3,4,5,6]
  results = [first_repeated_index(np.random.choice(dice, p=normal_dice, size=7))+1 for _ in range(n)]
  return {result: results.count(result) for result in possible_results}
  

if __name__ == "__main__":
  iterations = 100000
  final_dict = dice_problem(iterations)
  print(final_dict)
  print(sum(final_dict.values()))