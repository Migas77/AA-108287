from functools import cache

def fibonacci(n):
  global counter

  if n in [0,1]:
    return n
  else:
    counter += 1
    return fibonacci(n-1) + fibonacci(n-2)
  
@cache
def a_memoization_fibonacci(n):
  global counter

  if n in [0,1]:
    return n
  else:
    counter += 1
    return a_memoization_fibonacci(n-1) + a_memoization_fibonacci(n-2)
  

a,b = 0, 1
# 2
def m_memoization_fibonacci(n):
  global a,b,counter
  if n in [0,1]:
    return n
  res = a + b
  a = b
  b = n
  counter += 1
  return res


  

  
if __name__ == "__main__":
  n = 0
  print("{:^16}|{:^16}|{:^16}".format("n", "F(n)", "Adds(n)"))
  while True:
    counter = 0
    # print("{:^16}|{:^16}|{:^16}".format(n, fibonacci(n), counter))
    # print(n, a_memoization_fibonacci(n), counter)
    print(n, m_memoization_fibonacci(n), counter)
    if n == 10:
      break
    n += 1
