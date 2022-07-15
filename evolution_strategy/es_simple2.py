import numpy as np
import matplotlib.pyplot as plt



def evoluton_strategy(f, population_size, sigma, lr, initial_params, num_iters):


    num_parameters = len(initial_params)
    reward_per_iteration = np.zeros(num_iters)

    params = initial_params

    for t in range(num_iters):
        N = np.random.randn(population_size, num_parameters)
        R = np.zeros(population_size)

        for j in range(population_size):
            params_to_try = params + sigma*N[j]
            R[j] = f(params_to_try)

        mean = R.mean()
        std = R.std()
        A = (R - mean) / std
        reward_per_iteration[t] = mean
        params = params + lr / (population_size*sigma) * np.dot(N.T, A)

    return params, reward_per_iteration


def reward_function(params):
    X0 = params[0]
    X1= params[1]
    X2 = params[2]
    return -(X0**2 + 0.1*(X1-1)**2 + 0.5*(X2+3)**2)


if __name__ == '__main__':
    best_params, rewards = evoluton_strategy(f=reward_function, population_size=50, sigma=0.1, lr=1e-6,
                                             initial_params=np.random.randn(3), num_iters=500000)

    plt.plot(rewards)
    plt.show()

    print(f'Final params: {best_params}')


