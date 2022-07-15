import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


if __name__ == '__main__':
  grid = negative_grid()

  print('rewards:')
  print_values(grid.rewards, grid)

  policy = {}
  for s in grid.actions.keys():
      policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)

  print("initial policy:")
  print(policy, grid)

  V = {}
  states = grid.all_states()
  for s in states:
      if s in grid.actions:
          V[s] = np.random.random()
      else:
          V[s] = 0

  while True:
      while True:
          biggest_change = 0
          for s in states:
              old_v = V[s]
              if s in policy:
                  a = policy[s]
                  grid.set_state(s)
                  r = grid.move(a)
                  V[s] = r + GAMMA * V[grid.current_state()]
                  biggest_change = max(biggest_change, np.abs(old_v - V[s]))
          if biggest_change < SMALL_ENOUGH:
              break

      is_policy_converged = True
      for s in states:
          if s in policy:
              old_a = policy[s]
              new_a = None
              best_value = float('-inf')

              for a in ALL_POSSIBLE_ACTIONS:
                  grid.set_state(s)
                  r = grid.move(a)
                  v = r + GAMMA * V[grid.current_state()]
                  if v > best_value:
                      new_a = a
                      best_value = v
                      policy[s] = new_a
              if new_a != old_a:
                  is_policy_converged = False
      if is_policy_converged:
          break



  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)




















