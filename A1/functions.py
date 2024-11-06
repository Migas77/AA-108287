def f1(n):
  i,r=0
  for i in range(n):
    r += i
  return r


def f2(n):
  i,j,r=0;
  for i in range(n):
    for j in range(n):
      r += 1
  return r
  

def f3():
  i,j,r=0
  for i in range(n):
    for j in range(i,n+1):
      r += 1
  return r

def f4():
  pass




if __name__ == "main":
  
  print("f1")
  f1()

  print("\nf2")
  f2()

  print("\nf3")
  f3()

  print("\nf4")
  f4()