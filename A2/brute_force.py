def pow1(a, b):
  global iterative_counter
  # iterative b>=0
  result = 1
  for i in range(b):
    result *= a
    iterative_counter += 1
  return result

def pow2(a,b):
  global recursive_counter
  # recursive b>=0
  if b == 0:
    return 1
  recursive_counter += 1
  return a * pow2(a, b-1)

if __name__ == "__main__":
  a = 2
  print(f"a={a}")
  print("{:^16}|{:^16}|{:^16}".format("n", "a^n(i r)", "#Mults(i r)"))
  for b in range(10):
    iterative_counter = 0
    recursive_counter = 0
    print("{:^16}|{:^8}{:^8}|{:^8}{:^8}".format(b,pow1(a, b), pow2(a, b), iterative_counter, recursive_counter))



