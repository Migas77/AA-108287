import numpy as np

# Heads -> True
def dice_problem(n):
  # [True, False]
  # [Heads, Tails]
  possible_results_coin = [i for i in range(7)]
  coins = [True, False]
  balanced_coin = [1/2, 1/2]
  biased_coin = [2/3, 1/3]
  possible_results_dice = [2,3,4,5,6,7,8,9,10,11,12]
  dice_faces = [1,2,3,4,5,6]
  balanced_dice = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
  biased_dice = [2/7, 1/7, 1/7, 1/7, 1/7, 1/7]
  results = [sum(np.random.choice(coins, p=balanced_coin, size=times)) for times in [np.random.choice(dice_faces, p=balanced_dice, size=1) for _ in range(n)]]
  return {result: results.count(result) for result in possible_results_coin}
  

if __name__ == "__main__":
  iterations = 100000
  final_dict = dice_problem(iterations)
  print(final_dict)
  print({key: 100*value/iterations for key, value in final_dict.items()})
  print(sum(final_dict.values()))