import gym
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer

max_iter = 5e5
epsilon = 1e-2
gamma = 0.9

def compute_new_value(env, V_old, s):
    val_exp = np.zeros(env.action_space.n)

    for a in range(env.action_space.n):
        future_value = 0

        for prob, s_prim, reward, _ in env.P[s][a]:
            future_value += prob * V_old[s_prim]

        val_exp[a] = reward + gamma * future_value

    V = np.max(val_exp)

    return V

def standard_vi(env):
    V_old = np.zeros(env.observation_space.n)  # Value function from previous iteration
    V_new = np.zeros(env.observation_space.n)  # Value function for current iteration

    iter = 0
    differences = []

    while iter < max_iter:
        max_diff = 0

        V_old = V_new.copy()

        for s in range(env.observation_space.n):
            V_new[s] = compute_new_value(env, V_old, s)
            
            max_diff = max(max_diff, abs(V_old[s] - V_new[s]))  
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            differences.append(abs(V_old[s] - V_new[s]))
                #end !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            iter = iter + 1

        if max_diff < epsilon:
            break

    print("Total iterations spent: {}".format(iter))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot([i for i in range(iter)], differences, label="||V_(k+1)(s) - V_k(s)||")
    ax.legend()
    ax.get_legend().set_title("Standard Value iteration covergence")
    plt.show()
    return V_new


def gauss_seidel_vi(env, V_star, gamma=0.9, epsilon=1e-2, max_iter=int(5e5)):
    V = np.zeros(env.observation_space.n)
    differences = []  # Store ||V_k - V^*||_2
    iter = 0

    while iter < max_iter:
        max_diff = 0
        for s in range(env.observation_space.n):
            V_old = V[s]
            # Update V[s] in place using the most recent values
            V[s] = compute_new_value(env, V, s)
            max_diff = max(max_diff, abs(V_old - V[s]))
            
            # Track ||V_k - V^*||_2
            differences.append(np.linalg.norm(V - V_star, ord=2))
            
            iter += 1

        if max_diff < epsilon:
            break

    print("Total iterations spent: {}".format(iter))
    
    # Plot convergence
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(range(iter), differences, label="||V_k - V^*||_2")
    ax.legend()
    ax.get_legend().set_title("Gauss-Seidel Value Iteration Convergence")
    plt.show()

    return differences

def prioritized_sweeping_vi(env, V_star):
    V = np.zeros(env.observation_space.n)  # V_0(s)
    H = np.zeros(env.observation_space.n)  # H_0(s)

    differences = []  # Store ||V_k - V^*||_2
    iter = 0

    # Compute initial Bellman errors
    for s in range(env.observation_space.n):
        H[s] = abs(compute_new_value(env, V, s) - V[s])

    while iter < max_iter:
        error = 0

        s_k = np.argmax(H)
        bellman_error = H[s_k]

        # If the highest priority is below the threshold, stop
        if bellman_error < epsilon:
            break
        V_old = V[s_k]
        V[s_k] = compute_new_value(env, V, s_k)

        for s in range(env.observation_space.n):
            H[s] = abs(compute_new_value(env, V, s) - V[s]) 

        # Track ||V_k - V^*||_2
        error = max(error, abs(V_old - V[s_k]))

        differences.append(np.linalg.norm(V - V_star, ord=2))
        iter += 1

        if error < epsilon:
            break

    print("Total iterations spent: {}".format(iter))
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(iter), differences, label="||V_k - V*||_2")
    plt.xlabel("Iterations")
    plt.ylabel("||V_k - V*||_2")
    plt.title("Prioritized Sweeping Value Iteration Convergence")
    plt.legend()
    plt.grid()
    plt.show()

    return differences

def policy_iteration(env, V_star, gamma=0.9, epsilon=1e-2, max_iter=int(1e4)):
    # Initialize random policy
    policy = np.random.randint(0, env.action_space.n, size=env.observation_space.n)
    V = np.zeros(env.observation_space.n)  # Value function
    differences = []  # Store ||V_k - V^*||_2
    iter = 0

    while iter < max_iter:
        # Policy Evaluation
        while True:
            delta = 0
            for s in range(env.observation_space.n):
                V_old = V[s]
                a = policy[s]
                # for prob, s_prim, reward, _ in env.P[s][a]:
                    # V[s] += prob * V[s_prim]
                    # print(V[s])

                # V[s] = reward + gamma * V[s]
                V[s] = sum(prob * (reward + gamma * V[s_prim]) for prob, s_prim, reward, _ in env.P[s][a])
                delta = max(delta, abs(V_old - V[s]))

            if delta < epsilon:
                break

        # Track ||V_k - V^*||_2
        differences.append(np.linalg.norm(V - V_star, ord=2))

        # Policy Improvement
        policy_stable = True
        for s in range(env.observation_space.n):
            old_action = policy[s]
            action_values = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                # action_values[a] = sum(prob * (reward + gamma * V[s_prim])
                                    # for prob, s_prim, reward, _ in env.P[s][a])
                
                for prob, s_prim, reward, _ in env.P[s][a]:
                    action_values[a] += prob * V[s_prim]

                action_values[a] = reward + gamma * action_values[a]

            policy[s] = np.argmax(action_values)
            if old_action != policy[s]:
                policy_stable = False

        iter += 1

        # Check for convergence
        if policy_stable:
            break

    return iter, differences

def run_policy_iteration_instantiations(env, V_star):
    total_iterations = 0
    all_differences = []

    for i in range(5):
        print(f"Policy Iteration {i + 1}:")
        iterations, differences = policy_iteration(env, V_star)
        print(f"Number of iterations: {iterations}")
        total_iterations += iterations
        all_differences.append(differences)

    avg_iterations = total_iterations / 5
    print(f"Average iterations over 5 instantiations: {avg_iterations}")

    # Plot convergence for the last instantiation
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(all_differences[-1])), all_differences[-1], label="||V_k - V*||_2")
    plt.xlabel("Iterations")
    plt.ylabel("||V_k - V*||_2")
    plt.title("Policy Iteration Convergence (Last Instantiation)")
    plt.legend()
    plt.grid()
    plt.show()

    return avg_iterations


#env = gym.make("Taxi-v3")
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

V_star = standard_vi(env=env)
gs_differences = gauss_seidel_vi(env=env, V_star=V_star)

avg_iterations = run_policy_iteration_instantiations(env, V_star)

#ps_differences = prioritized_sweeping_vi(env, V_star)






