import numpy as np


def first_repeated_index(lst):
  seen = {}
  for index, element in enumerate(lst):
    if element in seen:
      return index
    seen[element] = index  # Store the index of the element's first occurrence
  return -1

# Heads -> True
def birthday_paradox(n):
  possible_results = [i for i in range(367)]
  normal_dice = [1/365 for i in range(365)]
  days = [i+1 for i in range(365)]
  results = [first_repeated_index(np.random.choice(days, p=normal_dice, size=366))+1 for _ in range(n)]
  return {result: results.count(result) for result in possible_results}
  

if __name__ == "__main__":
  iterations = 1000000
  final_dict = birthday_paradox(iterations)
  for k,v in sorted(final_dict.items()):
    print(k, 100*v/iterations)
  print(sum(final_dict.values()))


# Resposta 18 pessoas necessárias para ser superior a 50%