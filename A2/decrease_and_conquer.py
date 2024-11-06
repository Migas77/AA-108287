def my_pow(a, b):
  global counter
  if b == 0:
    return 1
  if b == 1:
    return a
  b2 = b // 2
  pre_result = my_pow(a,b2)

  if b % 2 == 0:
    counter += 1
    return pre_result * pre_result
  else:
    counter += 2
    return a * pre_result * pre_result


if __name__ == "__main__":
  a = 2
  print(f"a={a}")
  print("{:^16}|{:^16}|{:^16}".format("n", "a^n", "#Mults"))
  for b in range(10):
    counter = 0
    print("{:^16}|{:^16}|{:^16}".format(b,my_pow(a,b),counter))



