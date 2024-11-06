# Can move forward by 1 meter, or 2 meters, or 3 meters
# In how many ways can it move a distance of n meters ?

def linear_robot_recursive(n):
  global counter
  if n == 1:
    return 1
  elif n == 2:
    return 2
  elif n == 3:
    return 4
  else:
    counter += 2
    return linear_robot_recursive(n-1) + linear_robot_recursive(n-2) + linear_robot_recursive(n-3)
  
if __name__ == "__main__":
  n = 1
  print("{:^16}|{:^16}|{:^16}".format("n", "F(n)", "Adds(n)"))
  while True:
    counter = 0
    # print("{:^16}|{:^16}|{:^16}".format(n, fibonacci(n), counter))
    # print(n, a_memoization_fibonacci(n), counter)
    print(n, linear_robot_recursive(n), counter)
    if n == 40:
      break
    n += 1