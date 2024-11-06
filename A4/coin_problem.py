def coin_problem_with_heuristic(coins):
  amount = 0

  while len(coins) > 0:
    i = coins.index(max(coins))
    amount += coins[i]
    start = max(0, i - 1)
    end = min(len(coins), i + 2)
    del coins[start:end]
  return amount
    

def coin_problem_with_dp_and_recursive(coins,n):  
  global dp

  if n == 0:
    return 0
  if n == 1:
    return coins[1]
  return max(dp[n-1], coins[n] + dp[n+2])
  






if __name__ == "__main__":
  coins = [5,1,2,10,6,2]
  print(coin_problem_with_heuristic(coins))

  global dp
  coins = [5,1,2,10,6,2]
  dp = [0 for i in range(len(coins))]
  print(len(coins))
  print(dp)
  dp[0] = 0
  dp[1] = coins[1]
  print(coin_problem_with_dp_and_recursive([None] + coins, len(coins)))
  