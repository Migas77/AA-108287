import numpy as np

# Heads -> True
def dice_problem(n):
  # [True, False]
  # [Heads, Tails]
  possible_results = [
    (True, 0), (True, 1), (True, 2), (True, 3), (True, 4), (True, 5), (True, 6),
    (False, 0), (False, 1), (False, 2), (False, 3), (False, 4), (False, 5), (False, 6),
  ]
  colors = [True, False]
  balanced_colors = [1/2, 1/2]
  possible_results_dice = [2,3,4,5,6,7,8,9,10,11,12]
  dice_faces = [1,2,3,4,5,6]
  balanced_dice = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
  biased_dice = [2/7, 1/7, 1/7, 1/7, 1/7, 1/7]
  results = [
    (np.random.choice(colors, p=balanced_colors, size=1), np.random.choice(dice_faces, p=balanced_dice, size=1))
    for _ in range(n)
  ]
  return {result: results.count(result) for result in possible_results}
  

if __name__ == "__main__":
  iterations = 100000
  final_dict = dice_problem(iterations)
  print(final_dict)
  print({key: 100*value/iterations for key, value in final_dict.items()})
  print(sum(final_dict.values()))
  