import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def random_choice(a):
    p = np.random.random()
    if p < 0.5:
        return a
    else:
        tmp = list(ALL_POSSIBLE_ACTIONS)
        tmp.remove(a)
        return np.random.choice(tmp)

def play_game(grid, policy):
    start_states = list(grid.actions.keys())
    start_idx = np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])
    s = grid.current_state()
    states_and_rewards = [(s, 0)]
    while not grid.game_over():
        a = random_choice(policy[s])
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))

    G = 0
    states_and_returns = []
    first = True
    for s, r in reversed(states_and_rewards):
        if first:
            first = False
        else:
            states_and_returns.append((s, G))
        G = r + GAMMA * G

    states_and_returns.reverse()
    return  states_and_returns


if __name__ == '__main__':
    grid = standard_grid()

    print("rewards:")
    print_values(grid.rewards, grid)

    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'U',
        (2, 1): 'L',
        (2, 2): 'U',
        (2, 3): 'L',
    }

    V = {}
    returns = {}
    states = grid.all_states()

    for s in states:
        if s in grid.actions.keys():
            returns[s] = []
        else:
            V[s] = 0

    for t in range(5000):
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        for s, G in states_and_returns:
            if s not in seen_states:
                seen_states.add(s)
                returns[s].append(G)
                V[s] = np.mean(returns[s])

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
