import gym
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer as timer
import os

# Create directory for saving plots if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

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

def standard_vi(env, env_name):
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
            differences.append(abs(V_old[s] - V_new[s]))
            iter = iter + 1

        if max_diff < epsilon:
            break

    print("Standard Value Iteration - Total iterations spent: {}".format(iter))
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot([i for i in range(len(differences))], differences, label="||V_(k+1)(s) - V_k(s)||")
    ax.legend()
    ax.get_legend().set_title("Standard Value Iteration Convergence")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Absolute Difference")
    
    # Save the plot
    plt.savefig(f'plots/{env_name}_standard_vi.png')
    plt.show()
    
    return V_new


def gauss_seidel_vi(env, V_star, env_name, gamma=0.9, epsilon=1e-2, max_iter=int(5e5)):
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

    print("Gauss-Seidel Value Iteration - Total iterations spent: {}".format(iter))
    
    # Plot convergence
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(range(len(differences)), differences, label="||V_k - V^*||_2")
    ax.legend()
    ax.get_legend().set_title("Gauss-Seidel Value Iteration Convergence")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("L2 Norm Difference from V*")
    
    # Save the plot
    plt.savefig(f'plots/{env_name}_gauss_seidel_vi.png')
    plt.show()

    return differences

def prioritized_sweeping_vi(env, V_star, env_name, gamma=0.9, epsilon=1e-2, max_iter=int(5e5)):
    V = np.zeros(env.observation_space.n)  # V_0(s)
    H = np.zeros(env.observation_space.n)  # H_0(s) - priority queue

    differences = []  # Store ||V_k - V^*||_2
    iter = 0

    # Compute initial Bellman errors
    for s in range(env.observation_space.n):
        H[s] = abs(compute_new_value(env, V, s) - V[s])

    while iter < max_iter:
        # Select state with highest priority
        s_k = np.argmax(H)
        bellman_error = H[s_k]

        # If the highest priority is below the threshold, stop
        if bellman_error < epsilon:
            break
            
        V_old = V[s_k]
        V[s_k] = compute_new_value(env, V, s_k)

        # Update priorities for all states
        for s in range(env.observation_space.n):
            H[s] = abs(compute_new_value(env, V, s) - V[s]) 

        # Track ||V_k - V^*||_2
        differences.append(np.linalg.norm(V - V_star, ord=2))
        iter += 1

    print("Prioritized Sweeping Value Iteration - Total iterations spent: {}".format(iter))
    
    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(differences)), differences, label="||V_k - V*||_2")
    plt.xlabel("Iterations")
    plt.ylabel("||V_k - V*||_2")
    plt.title("Prioritized Sweeping Value Iteration Convergence")
    plt.legend()
    plt.grid()
    
    # Save the plot
    plt.savefig(f'plots/{env_name}_prioritized_sweeping_vi.png')
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
                action_values[a] = 0
                for prob, s_prim, reward, _ in env.P[s][a]:
                    action_values[a] += prob * (reward + gamma * V[s_prim])

            policy[s] = np.argmax(action_values)
            if old_action != policy[s]:
                policy_stable = False

        iter += 1

        # Check for convergence
        if policy_stable:
            break

    return iter, differences

def run_policy_iteration_instantiations(env, V_star, env_name):
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
    
    # Save the plot
    plt.savefig(f'plots/{env_name}_policy_iteration.png')
    plt.show()
    
    # Also plot all instantiations together
    plt.figure(figsize=(10, 6))
    for i, diffs in enumerate(all_differences):
        plt.plot(range(len(diffs)), diffs, label=f"Instance {i+1}")
    plt.xlabel("Iterations")
    plt.ylabel("||V_k - V*||_2")
    plt.title("Policy Iteration Convergence (All Instantiations)")
    plt.legend()
    plt.grid()
    
    # Save the plot
    plt.savefig(f'plots/{env_name}_policy_iteration_all.png')
    plt.show()

    return avg_iterations

def run_experiment(env_name):
    print(f"\n=== Running experiment for {env_name} ===\n")
    
    if env_name == "Taxi-v3":
        env = gym.make("Taxi-v3")
    else:
        env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    
    print(f"State space size: {env.observation_space.n}")
    print(f"Action space size: {env.action_space.n}")
    
    print("\nStep 1: Computing V* using standard Value Iteration")
    start_time = timer()
    V_star = standard_vi(env=env, env_name=env_name)
    end_time = timer()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    print("\nStep 2a: Running Gauss-Seidel Value Iteration")
    start_time = timer()
    gs_differences = gauss_seidel_vi(env=env, V_star=V_star, env_name=env_name)
    end_time = timer()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    print("\nStep 2b: Running Prioritized Sweeping Value Iteration")
    start_time = timer()
    ps_differences = prioritized_sweeping_vi(env=env, V_star=V_star, env_name=env_name)
    end_time = timer()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    print("\nStep 3: Running Policy Iteration (5 instantiations)")
    start_time = timer()
    avg_iterations = run_policy_iteration_instantiations(env, V_star, env_name)
    end_time = timer()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    # Compare all methods
    plt.figure(figsize=(12, 8))
    if len(gs_differences) > 0 and len(ps_differences) > 0:
        # Sample points to match lengths for fair comparison
        gs_sample = np.linspace(0, len(gs_differences)-1, min(100, len(gs_differences)))
        ps_sample = np.linspace(0, len(ps_differences)-1, min(100, len(ps_differences)))
        
        plt.plot(gs_sample, [gs_differences[int(i)] for i in gs_sample], 'b-', label="Gauss-Seidel VI")
        plt.plot(ps_sample, [ps_differences[int(i)] for i in ps_sample], 'r-', label="Prioritized Sweeping VI")
        plt.xlabel("Sampled Iterations")
        plt.ylabel("||V_k - V*||_2")
        plt.title(f"Convergence Comparison for {env_name}")
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(f'plots/{env_name}_comparison.png')
        plt.show()

# Run experiments
print("==== STARTING EXPERIMENTS ====")

# Run experiment for Taxi-v3
run_experiment("Taxi-v3")

# Run experiment for FrozenLake-v1
run_experiment("FrozenLake-v1")

print("==== ALL EXPERIMENTS COMPLETED ====")
print(f"All plots have been saved to the 'plots' directory")