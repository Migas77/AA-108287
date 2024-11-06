import numpy as np

# Heads -> True
def dice_problem(n):
  # Red, Green
  possible_results = [(i,j) for i in range(1,7) for j in range(1,7)]
  dice_faces = [1,2,3,4,5,6]
  balanced_dice = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
  results = [
    (np.random.choice(dice_faces, p=balanced_dice, size=1), np.random.choice(dice_faces, p=balanced_dice, size=1),)
    for _ in range(n)
  ]
  return {result: results.count(result) for result in possible_results}
  

if __name__ == "__main__":
  iterations = 100000
  final_dict = dice_problem(iterations)
  print(final_dict)
  # losing
  print(sum(count for key, count in final_dict.items() if key[0] >= key[1]))
  print(sum(count for key, count in final_dict.items()))
  