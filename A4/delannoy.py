def delannoy_recursive(i,j):
  if i == 0 or j == 0:
    return 1
  return delannoy_recursive(i-1, j) + delannoy_recursive(i, j-1) + delannoy_recursive(i-1, j-1)




def delannoy_dynamic_2D_array(m,n):
  dp = [ [None]*(n+1) for k in range(m+1)]

  # Return the (i,j) Delannoy Number.
  for i in range(m+1):
    dp[i][0] = 1
  for j in range(n+1):
    dp[0][j] = 1

  for i in range(1,m+1):
    for j in range(1,n+1):
      dp[i][j] = dp[i-1][j] + dp[i][j-1] + dp[i-1][j-1]

  return dp[m][n]

  

print(delannoy_recursive(10,10))  
print(delannoy_dynamic_2D_array(10,10))

# https://www.geeksforgeeks.org/delannoy-number/