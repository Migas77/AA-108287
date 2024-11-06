def my_pow(a, b):
  global counter
  if b == 0:
    return 1
  if b == 1:
    return a
  # b1 = b//2; b2 = b - b1
  b1 = b // 2; b2 = (b+1) // 2
  counter += 1
  return my_pow(a,b1) * my_pow(a,b2)


if __name__ == "__main__":
  a = 2
  print(f"a={a}")
  print("{:^16}|{:^16}|{:^16}".format("n", "a^n", "#Mults"))
  for b in range(10):
    counter = 0
    print("{:^16}|{:^16}|{:^16}".format(b,my_pow(a,b),counter))



