import numpy as np

# Heads -> True
def coin_problem(n):
  possible_results = [i for i in range(4)]
  # [True, False]
  # [Heads, Tails]
  normal_coin = [1/2, 1/2]
  biased_coin = [2/3, 1/3]
  coins = [True, False]
  results = []
  for _ in range(n):
    coin_set = np.random.choice(coins, p=normal_coin, size=3)
    if coin_set[0] == coin_set[1]:
      results.append(2)
    else:
      results.append(3)

  return {result: results.count(result) for result in possible_results}
  

if __name__ == "__main__":
  iterations = 100000
  final_dict = coin_problem(iterations)
  print(final_dict)
  print(final_dict[0] + final_dict[1] + final_dict[2] + final_dict[3])