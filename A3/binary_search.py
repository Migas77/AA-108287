import random

dict = dict()

def binary_search(lst, low, high, x):
  global dict, counter
  counter += 1

  # print(f"low: {low}, high: {high}, counter: {counter}")

  if high >= low:
    middle = (high + low) // 2
    if lst[middle] == x:
      dict[counter] = dict.get(counter, 0) + 1
      return x
    elif lst[middle] > x:
      return binary_search(lst, low, middle - 1, x)
    else:
      return binary_search(lst, middle + 1, high, x)
  else:
    return -1

def game(lst, number):
  global counter
  counter = 0
  binary_search(lst, 0, len(lst) - 1, number)


if __name__ == "__main__":
  n = 100000
  a = 1
  b = 100
  lst = list(range(a, b + 1))

  print(f"Simulation: playing the higher-lower game {n} times.\nThe interval of values is [{a},{b}]")
  
  for i in range(n):
    game(lst, random.randint(a, b))

  for attempts, count in sorted(dict.items()):
    print(f"Number of attempts: {attempts}: {count} - {100*count/n}%")

  print(f"MIN - The smallest number of attempts = {min(dict.keys())}")
  print(f"MEDIAN - The median number of attempts = {sorted([k for k in dict for v in range(dict[k])])[sum(v for v in dict.values()) // 2]}")
  print(f"MEAN - The average number of attempts = {sum(k*v for k, v in dict.items()) / n}")
  print(f"MAX - The largest number of attempts = {max(dict.keys())}")