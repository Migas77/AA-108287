import numpy as np

boys_and_girls_domain = 51 * ['b'] + 49 * ['g']

def monte_carlo_approximation(num_points):
  inside_counter = 0
  for i in range(num_points):
    two_children = np.random.choice(boys_and_girls_domain, size=2)
    if (two_children == ['g', 'g']).all():
      inside_counter += 1
    
  return (inside_counter/num_points) * len(boys_and_girls_domain)
    

if __name__ == "__main__":
  # probability of having two daughters
  # the result should be 24%
  for iter in [1,10,100,1000,10000,100000,1000000]:
    print(monte_carlo_approximation(iter))


