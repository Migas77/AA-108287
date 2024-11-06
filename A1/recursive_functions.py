def r3(n):
  if n==0:
    return 0
  return 1 + 2 * r3(n-1)

def r4(n):
  if n==0:
    return 0
  return 1 + r4(n-1) + r4(n-1)

if __name__ == "__main__":
  n = int(input("Enter n: "))

  print(r3(n))
  print(r4(n))