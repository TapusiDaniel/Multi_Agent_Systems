import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import os 
from tqdm import tqdm 

# --- Task Configuration ---
ENV_NAME = "FrozenLake-v1"
MAP_NAME = "4x4"
IS_SLIPPERY = True 

NUM_EPISODES = 8000
NUM_REPETITIONS = 20
GAMMA = 0.99
ALPHAS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
N_VALUES = [2, 3, 4] 

# Epsilon Decay Parameters
EPS_START = 0.75
EPS_END = 0.001
DECAY_END_EPISODE = int(0.3 * NUM_EPISODES)

# Value Iteration Parameters
VI_THETA = 1e-9
VI_MAX_ITER = 10000 

# --- Plot Saving Configuration ---
PLOT_DIR = "plots" 
PLOT_FILENAME = f"rmse_vs_alpha_{ENV_NAME}_{MAP_NAME}_ep{NUM_EPISODES}.png"

# --- Helper Functions ---
def get_epsilon(episode):
    """Calculates epsilon value with linear decay."""
    if episode < DECAY_END_EPISODE:
        # Ensure division is float division
        decay_fraction = episode / float(DECAY_END_EPISODE)
        return EPS_START + (EPS_END - EPS_START) * decay_fraction
    else:
        return EPS_END

def choose_action_eps_greedy(Q, state, epsilon, action_space_n):
    """Chooses an action using epsilon-greedy policy."""
    if random.uniform(0, 1) < epsilon:
        return random.choice(range(action_space_n))
    else:
        # np.argmax breaks ties arbitrarily by choosing the first max found
        # Handle potential multiple max values if needed (not strictly necessary here)
        # best_actions = np.flatnonzero(Q[state, :] == np.max(Q[state, :]))
        # return random.choice(best_actions)
        return np.argmax(Q[state, :])

def calculate_rmse(Q, V_star):
    """Calculates RMSE between max_a Q(s, a) and V_star."""
    if Q is None or V_star is None:
         print("Warning: Q or V_star is None in calculate_rmse")
         return float('inf') # Return infinity if inputs are invalid
    if Q.shape[0] != V_star.shape[0]:
        raise ValueError(f"Shape mismatch: Q states {Q.shape[0]}, V_star states {V_star.shape[0]}")

    V_estimated = np.max(Q, axis=1)
    if V_estimated.shape != V_star.shape:
         # This should ideally not happen if the previous check passed, but good failsafe
         raise ValueError(f"Shape mismatch after max: V_estimated {V_estimated.shape}, V_star {V_star.shape}")

    mse = np.mean((V_estimated - V_star) ** 2)
    rmse = np.sqrt(mse)
    return rmse

# --- Value Iteration (Ground Truth V*) ---

def value_iteration(env, gamma, theta, max_iterations):
    """Computes the optimal state-value function V* using Value Iteration."""
    print("Running Value Iteration...")
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    V = np.zeros(num_states)
    iterations = 0
    while iterations < max_iterations:
        delta = 0
        V_old = V.copy()
        for s in range(num_states):
            q_sa = np.zeros(num_actions)
            for a in range(num_actions):
                # env.P[s][a] is a list of (prob, next_state, reward, done) tuples
                for prob, next_state, reward, _ in env.P[s][a]:
                     # Use V_old for the bootstrap value from the previous iteration
                    q_sa[a] += prob * (reward + gamma * V_old[next_state])
            V[s] = np.max(q_sa) # Bellman optimality update
            delta = max(delta, abs(V[s] - V_old[s]))

        iterations += 1
        if delta < theta:
            print(f"Value Iteration converged after {iterations} iterations.")
            break
    if iterations == max_iterations:
         print(f"Value Iteration reached max iterations ({max_iterations}) without converging below theta={theta}.")

    return V

# --- TD Learning Algorithms (with Gym API fixes) ---

def q_learning_run(env, alpha, gamma, num_episodes):
    """Runs Q-Learning algorithm."""
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    action_space_n = env.action_space.n

    for episode in range(num_episodes):
        state_info = env.reset()
        state = state_info[0] if isinstance(state_info, tuple) else state_info # Get state from tuple if needed
        done = False
        epsilon = get_epsilon(episode)

        while not done:
            action = choose_action_eps_greedy(Q, state, epsilon, action_space_n)

            # --- Gym API Fix ---
            step_result = env.step(action)
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated # Episode ends if terminated or truncated
            # --- End Fix ---

            # Handle potential tuple return for next_state (less common post-reset, but safe)
            next_state = next_state[0] if isinstance(next_state, tuple) else next_state

            # Q-Learning Update
            best_next_action_value = np.max(Q[next_state, :])
            td_target = reward + gamma * best_next_action_value
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
    return Q

def sarsa_run(env, alpha, gamma, num_episodes):
    """Runs SARSA algorithm."""
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    action_space_n = env.action_space.n

    for episode in range(num_episodes):
        state_info = env.reset()
        state = state_info[0] if isinstance(state_info, tuple) else state_info
        done = False
        epsilon = get_epsilon(episode)
        action = choose_action_eps_greedy(Q, state, epsilon, action_space_n)

        while not done:
            # --- Gym API Fix ---
            step_result = env.step(action)
            next_state, reward, terminated, truncated, info = step_result
            done = terminated or truncated
            # --- End Fix ---

            next_state = next_state[0] if isinstance(next_state, tuple) else next_state

            # Choose next_action using epsilon-greedy based on Q and next_state
            epsilon_next = get_epsilon(episode) # Could use the same epsilon or recalculate
            next_action = choose_action_eps_greedy(Q, next_state, epsilon_next, action_space_n)

            # SARSA Update (Standard form)
            td_target = reward + gamma * Q[next_state, next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            action = next_action
    return Q

def n_step_sarsa_run(env, n, alpha, gamma, num_episodes):
    """Runs n-step SARSA algorithm."""
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    action_space_n = env.action_space.n
    # Precompute powers of gamma for efficiency
    gamma_powers = np.array([gamma**i for i in range(n + 1)])

    for episode in range(num_episodes):
        state_info = env.reset()
        state = state_info[0] if isinstance(state_info, tuple) else state_info
        epsilon = get_epsilon(episode)
        action = choose_action_eps_greedy(Q, state, epsilon, action_space_n)

        # Store trajectory: S_0, A_0, R_1, S_1, A_1, R_2, ...
        states = [state]
        actions = [action]
        rewards = [0.0] # Placeholder R_0, actual rewards start at index 1

        T = float('inf') # Terminal time step index
        t = 0 # Current time step index
        while True:
            if t < T:
                # --- Gym API Fix ---
                step_result = env.step(actions[t])
                next_state, reward, terminated, truncated, info = step_result
                is_done = terminated or truncated # Use a distinct variable
                # --- End Fix ---

                next_state = next_state[0] if isinstance(next_state, tuple) else next_state

                states.append(next_state)
                rewards.append(float(reward)) # Ensure reward is float

                if is_done:
                    T = t + 1 # Set terminal time index
                else:
                    # Choose next action if not done
                    epsilon_next = get_epsilon(episode)
                    next_action = choose_action_eps_greedy(Q, next_state, epsilon_next, action_space_n)
                    actions.append(next_action)

            # tau is the time step whose estimate is being updated
            tau = t - n + 1

            if tau >= 0: # Start updates once we have n steps or termination
                G = 0.0
                # Calculate n-step return G_t:t+n
                # Sum rewards R_{tau+1} to R_{min(tau+n, T)}
                limit = min(tau + n, T)
                for i in range(tau + 1, limit + 1):
                    # Access reward R_i using index i
                    if i < len(rewards):
                        G += gamma_powers[i - tau - 1] * rewards[i]
                    # else: # Should not happen if T is set correctly
                    #     print(f"Warning: Reward index {i} out of bounds (len={len(rewards)}, T={T}, tau={tau})")

                # Add bootstrap term if tau+n < T
                if tau + n < T:
                    # Bootstrap value is Q(S_{tau+n}, A_{tau+n})
                    # Ensure indices are within bounds of the trajectory stored so far
                    if tau + n < len(states) and tau + n < len(actions):
                        bootstrap_state = states[tau + n]
                        bootstrap_action = actions[tau + n]
                        G += gamma_powers[n] * Q[bootstrap_state, bootstrap_action]
                    # else: # Should not happen if logic is correct
                    #     print(f"Warning: Bootstrap index {tau+n} out of bounds (len_S={len(states)}, len_A={len(actions)}, T={T}, tau={tau})")


                # Update Q(S_tau, A_tau)
                # Ensure tau is a valid index for the state and action being updated
                if tau < len(states) and tau < len(actions):
                    state_tau = states[tau]
                    action_tau = actions[tau]
                    td_error = G - Q[state_tau, action_tau]
                    Q[state_tau, action_tau] += alpha * td_error
                # else: # Should not happen
                #      print(f"Warning: Update index {tau} out of bounds (len_S={len(states)}, len_A={len(actions)}, T={T})")


            if tau == T - 1: # If the update was for the last time step before termination
                break # Exit the inner loop (episode finished)

            t += 1
            # Note: No need to append action here as it's handled within the 'if t < T' block

    return Q

# --- Main Experiment ---

# Create environment
# Use the registered name with render_mode='ansi' or 'rgb_array' if visual needed later
# For pure computation, no render_mode needed initially.
env = gym.make(ENV_NAME, map_name=MAP_NAME, is_slippery=IS_SLIPPERY)
print(f"Environment: {ENV_NAME} ({MAP_NAME}, Slippery: {IS_SLIPPERY})")
print(f"Observation Space: {env.observation_space.n}")
print(f"Action Space: {env.action_space.n}")
print("-" * 30)
print(f"Episodes per run: {NUM_EPISODES}")
print(f"Repetitions per alpha: {NUM_REPETITIONS}")
print(f"Gamma (Discount Factor): {GAMMA}")
print(f"Alpha (Learning Rate) values: {ALPHAS}")
print(f"Epsilon Decay: Start={EPS_START}, End={EPS_END}, Decay End Episode={DECAY_END_EPISODE}")
print(f"N-Step SARSA values (n): {N_VALUES}")
print("-" * 30)

# 1. Calculate Ground Truth V* using Value Iteration
V_star = value_iteration(env, GAMMA, VI_THETA, VI_MAX_ITER)
if V_star is not None:
    print(f"Ground Truth V* (first 5 states): {V_star[:5]}")
    print(f"Ground Truth V* (last 5 states): {V_star[-5:]}")
else:
    print("ERROR: Value Iteration failed to produce V*")
    exit() # Stop if ground truth is not available
print("-" * 30)

# 2. Run experiments and store results
# Dictionary structure: {alg_name: {alpha: [rmse_rep1, rmse_rep2, ...]}}
results = {
    "Q-Learning": {alpha: [] for alpha in ALPHAS},
    "SARSA": {alpha: [] for alpha in ALPHAS},
}
for n in N_VALUES:
    results[f"{n}-Step SARSA"] = {alpha: [] for alpha in ALPHAS}

# Use tqdm for progress bars if installed
try:
    from tqdm import tqdm
    alpha_iterator = tqdm(ALPHAS, desc="Alpha Progress")
except ImportError:
    print("Optional dependency 'tqdm' not found. Progress bars disabled.")
    print("Install with: pip install tqdm")
    alpha_iterator = ALPHAS

for alpha in alpha_iterator:
    print(f"\n--- Running for Alpha = {alpha:.2f} ---")
    # Optionally use tqdm for the inner loop too
    # rep_iterator = tqdm(range(NUM_REPETITIONS), desc=f" Reps (Alpha={alpha:.1f})", leave=False)
    rep_iterator = range(NUM_REPETITIONS)

    for rep in rep_iterator:
        # print(f"  Repetition {rep + 1}/{NUM_REPETITIONS}") # Can be noisy with tqdm

        # Run Q-Learning
        Q_q = q_learning_run(env, alpha, GAMMA, NUM_EPISODES)
        rmse_q = calculate_rmse(Q_q, V_star)
        results["Q-Learning"][alpha].append(rmse_q)

        # Run SARSA
        Q_s = sarsa_run(env, alpha, GAMMA, NUM_EPISODES)
        rmse_s = calculate_rmse(Q_s, V_star)
        results["SARSA"][alpha].append(rmse_s)

        # Run n-step SARSA for each n
        for n in N_VALUES:
            Q_n = n_step_sarsa_run(env, n, alpha, GAMMA, NUM_EPISODES)
            rmse_n = calculate_rmse(Q_n, V_star)
            results[f"{n}-Step SARSA"][alpha].append(rmse_n)

env.close() # Close the environment when done with all experiments

# 3. Calculate Average RMSE over repetitions
avg_rmse_results = {}
std_rmse_results = {} # Optional: Calculate standard deviation for error bars
for alg_name, alpha_data in results.items():
    avg_rmses = []
    std_rmses = []
    for alpha in ALPHAS:
        rep_rmses = alpha_data[alpha]
        if rep_rmses: # Ensure list is not empty
            avg_rmses.append(np.mean(rep_rmses))
            std_rmses.append(np.std(rep_rmses))
        else: # Handle case where results might be missing (should not happen ideally)
            avg_rmses.append(float('nan'))
            std_rmses.append(float('nan'))
    avg_rmse_results[alg_name] = avg_rmses
    std_rmse_results[alg_name] = std_rmses # Store std dev


print("\n--- Average RMSE Results (Lower is Better) ---")
for alg_name, avg_rmses in avg_rmse_results.items():
    print(f"{alg_name}:")
    for i, alpha in enumerate(ALPHAS):
        print(f"  Alpha={alpha:.1f}: {avg_rmses[i]:.4f} (+/- {std_rmse_results[alg_name][i]:.4f})") # Also show std dev

# 4. Plotting
plt.figure(figsize=(12, 8))

for alg_name, avg_rmses in avg_rmse_results.items():
    # Optionally plot with error bars (standard deviation)
    # plt.errorbar(ALPHAS, avg_rmses, yerr=std_rmse_results[alg_name], marker='o', linestyle='-', label=alg_name, capsize=3)
    plt.plot(ALPHAS, avg_rmses, marker='o', linestyle='-', label=alg_name)


plt.xlabel("Alpha (Learning Rate)")
plt.ylabel(f"Average RMSE over {NUM_REPETITIONS} Repetitions")
plt.title(f"Algorithm Performance on {ENV_NAME} ({MAP_NAME}) - {NUM_EPISODES} Episodes per Rep")
plt.legend(loc='best') # Adjust legend location if needed
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(ALPHAS) # Ensure all alpha values are shown on the x-axis
plt.ylim(bottom=0) # RMSE cannot be negative
plt.tight_layout() # Adjust layout to prevent labels overlapping

# 5. Save the plot
# Create the directory if it doesn't exist
os.makedirs(PLOT_DIR, exist_ok=True)
plot_path = os.path.join(PLOT_DIR, PLOT_FILENAME)

try:
    plt.savefig(plot_path)
    print(f"\nPlot saved successfully to: {plot_path}")
except Exception as e:
    print(f"\nError saving plot: {e}")


# 6. Display the plot
plt.show()

print("\nExperiment finished.")