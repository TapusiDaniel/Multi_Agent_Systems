import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict
from tqdm import tqdm
import os

# Create directories for saving results
os.makedirs('results', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Define actions
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]
ACTION_NAMES = ['Up', 'Right', 'Down', 'Left']

#########################
# Environment Classes
#########################

class GridWorld:
    """Base class for grid world environments"""
    def __init__(self, grid_type='A'):
        self.grid_type = grid_type
        self.height = 7
        self.width = 10
        self.start_pos = (3, 1)  # Row, Column (S position)
        self.goal_pos = (3, 8)   # Row, Column (G position)
        
        if grid_type == 'A':
            # Obstacles (grey cells) in column 6 (index 5), rows 2-5 (indices 1-4)
            self.obstacles = [(i, 5) for i in range(1, 5)]
        else:  # grid_type == 'B'
            self.obstacles = []
            # Wind strength for each column (0-indexed)
            self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    
    def reset(self):
        """Reset the environment and return the initial state"""
        return self.start_pos
    
    def get_next_state(self, state, action):
        """Compute the next state given current state and action"""
        row, col = state
        
        # Apply action
        if action == UP: row -= 1
        elif action == RIGHT: col += 1
        elif action == DOWN: row += 1
        elif action == LEFT: col -= 1
        
        # Ensure agent stays within horizontal boundaries first
        col = max(0, min(col, self.width - 1))
        
        # Apply wind for Gridworld B (after ensuring col is valid)
        if self.grid_type == 'B':
            row -= self.wind[col]  # Wind pushes upward (decreases row)
        
        # Now ensure agent stays within vertical boundaries
        row = max(0, min(row, self.height - 1))
        
        # Check for obstacles
        if (row, col) in self.obstacles:
            return state  # Return original state if hitting obstacle
        
        return (row, col)
    
    def get_reward(self, state, next_state):
        """Return the reward for a state transition"""
        if next_state == self.goal_pos:
            return 1  # Reward for reaching goal
        else:
            return -1  # Penalty for each step
    
    def is_terminal(self, state):
        """Check if the state is terminal"""
        return state == self.goal_pos
    
    def print_grid(self, agent_pos=None, policy=None):
        """Visualize the grid with optional agent position and policy"""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Mark obstacles
        for r, c in self.obstacles:
            grid[r][c] = '█'
        
        # Mark start and goal
        grid[self.start_pos[0]][self.start_pos[1]] = 'S'
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        
        # Mark agent position if provided
        if agent_pos:
            r, c = agent_pos
            if grid[r][c] == ' ':
                grid[r][c] = 'A'
            else:
                grid[r][c] = grid[r][c] + 'A'
        
        # Mark policy if provided
        if policy:
            for r in range(self.height):
                for c in range(self.width):
                    if grid[r][c] == ' ' and (r, c) in policy:
                        action = policy[(r, c)]
                        if action == UP: grid[r][c] = '↑'
                        elif action == RIGHT: grid[r][c] = '→'
                        elif action == DOWN: grid[r][c] = '↓'
                        elif action == LEFT: grid[r][c] = '←'
        
        # Print the grid
        print(f"Gridworld {self.grid_type}")
        for r in range(self.height):
            print('|' + '|'.join(grid[r]) + '|')
        
        # Print wind if Gridworld B
        if self.grid_type == 'B':
            print(' '.join([str(w) for w in self.wind]))


class MultiAgentGridWorld(GridWorld):
    """Extension of GridWorld for multi-agent scenario"""
    def __init__(self, num_agents=3):
        super().__init__('B')  # Use Gridworld B for multi-agent scenario
        self.num_agents = num_agents
    
    def reset(self):
        """Reset the environment and return initial states for all agents"""
        return [self.start_pos] * self.num_agents
    
    def step(self, states, actions):
        """Take a step for all agents and return new states and rewards"""
        next_states = [self.get_next_state(state, action) 
                      for state, action in zip(states, actions)]
        
        # Check which agents reached the goal
        at_goal = [state == self.goal_pos for state in next_states]
        
        # Calculate rewards based on collaborative goal achievement
        if any(at_goal):
            if all(at_goal):
                rewards = [10] * self.num_agents  # All agents reached goal simultaneously
            else:
                rewards = [-0.5] * self.num_agents  # Some agents reached goal
        else:
            rewards = [-1] * self.num_agents  # No agent reached goal
            
        terminal = all(at_goal)
        
        return next_states, rewards, terminal
    
    def print_grid(self, agent_positions=None):
        """Visualize the grid with multiple agents"""
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Mark obstacles
        for r, c in self.obstacles:
            grid[r][c] = '█'
        
        # Mark start and goal
        grid[self.start_pos[0]][self.start_pos[1]] = 'S'
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        
        # Mark agent positions if provided
        if agent_positions:
            for i, (r, c) in enumerate(agent_positions):
                if grid[r][c] == ' ':
                    grid[r][c] = str(i)
                else:
                    grid[r][c] = grid[r][c] + str(i)
        
        # Print the grid
        print("Multi-Agent Gridworld B")
        for r in range(self.height):
            print('|' + '|'.join(grid[r]) + '|')
        
        # Print wind
        print(' '.join([str(w) for w in self.wind]))


#########################
# Utility Functions
#########################

def epsilon_greedy(Q, state, epsilon, actions=ACTIONS):
    """Epsilon-greedy policy: choose random action with probability epsilon,
    otherwise choose the best action according to Q-values"""
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        # Find actions with the highest Q-value
        q_values = [Q[state][a] for a in actions]
        max_q = max(q_values)
        # Get all actions with the maximum Q-value
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        # Randomly select from the best actions (to break ties)
        return random.choice(best_actions)


def extract_policy(Q):
    """Extract the greedy policy from a Q-function"""
    policy = {}
    for state in Q:
        policy[state] = max(ACTIONS, key=lambda a: Q[state][a])
    return policy


def analyze_results(algorithms, grid_types, epsilons, alphas, episodes_list, metrics):
    """Analyze and summarize the experimental results"""
    print("\n===== ANALYSIS OF RESULTS =====\n")
    
    for grid_type in grid_types:
        print(f"\n----- GRIDWORLD {grid_type} -----\n")
        
        for algo in algorithms:
            print(f"\n{algo.upper()} RESULTS:")
            
            # Find best parameters based on average path length
            best_avg_steps = float('inf')
            best_params = None
            
            for epsilon in epsilons:
                for alpha in alphas:
                    key = (algo, grid_type, epsilon, alpha)
                    
                    if key in metrics:
                        avg_steps = metrics[key]['avg_final_steps']
                        convergence_episode = metrics[key]['convergence_episode']
                        
                        print(f"  ε={epsilon}, α={alpha}: "
                              f"Avg Steps={avg_steps:.2f}, "
                              f"Convergence Episode={convergence_episode}")
                        
                        if avg_steps < best_avg_steps:
                            best_avg_steps = avg_steps
                            best_params = (epsilon, alpha)
            
            if best_params:
                print(f"\n  Best Parameters for {algo} in Gridworld {grid_type}: "
                      f"ε={best_params[0]}, α={best_params[1]} with Avg Steps={best_avg_steps:.2f}")
            else:
                print(f"\n  No valid results found for {algo} in Gridworld {grid_type}")


#########################
# Algorithm Implementations
#########################

def sarsa(env, epsilon, alpha, gamma=1.0, max_episodes=500, eval_interval=10):
    """SARSA algorithm implementation"""
    Q = defaultdict(lambda: defaultdict(float))  # Initialize Q-values to 0
    steps_per_episode = []
    eval_steps = []
    
    for episode in range(max_episodes):
        state = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        steps = 0
        
        while not env.is_terminal(state):
            next_state = env.get_next_state(state, action)
            reward = env.get_reward(state, next_state)
            
            if env.is_terminal(next_state):
                # Terminal state has no next action
                Q[state][action] += alpha * (reward - Q[state][action])
            else:
                next_action = epsilon_greedy(Q, next_state, epsilon)
                # SARSA update
                Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
                action = next_action
            
            state = next_state
            steps += 1
            
            # Prevent infinite loops
            if steps > 1000:
                break
        
        steps_per_episode.append(steps)
        
        # Evaluate policy periodically with epsilon=0 (greedy policy)
        if (episode + 1) % eval_interval == 0:
            eval_steps.append(evaluate_policy(env, Q, num_episodes=10))
    
    return Q, steps_per_episode, eval_steps


def q_learning(env, epsilon, alpha, gamma=1.0, max_episodes=500, eval_interval=10):
    """Q-learning algorithm implementation"""
    Q = defaultdict(lambda: defaultdict(float))  # Initialize Q-values to 0
    steps_per_episode = []
    eval_steps = []
    
    for episode in range(max_episodes):
        state = env.reset()
        steps = 0
        
        while not env.is_terminal(state):
            action = epsilon_greedy(Q, state, epsilon)
            next_state = env.get_next_state(state, action)
            reward = env.get_reward(state, next_state)
            
            # Q-learning update
            if env.is_terminal(next_state):
                # Terminal state has no future reward
                Q[state][action] += alpha * (reward - Q[state][action])
            else:
                # Use max Q-value for next state (regardless of policy)
                next_max_q = max(Q[next_state][a] for a in ACTIONS)
                Q[state][action] += alpha * (reward + gamma * next_max_q - Q[state][action])
            
            state = next_state
            steps += 1
            
            # Prevent infinite loops
            if steps > 1000:
                break
        
        steps_per_episode.append(steps)
        
        # Evaluate policy periodically
        if (episode + 1) % eval_interval == 0:
            eval_steps.append(evaluate_policy(env, Q, num_episodes=10))
    
    return Q, steps_per_episode, eval_steps


def double_q_learning(env, epsilon, alpha, gamma=1.0, max_episodes=500, eval_interval=10):
    """Double Q-learning algorithm implementation"""
    # Initialize two Q-value functions
    Q1 = defaultdict(lambda: defaultdict(float))
    Q2 = defaultdict(lambda: defaultdict(float))
    steps_per_episode = []
    eval_steps = []
    
    for episode in range(max_episodes):
        state = env.reset()
        steps = 0
        
        while not env.is_terminal(state):
            # Compute combined Q-values for epsilon-greedy policy
            Q_combined = defaultdict(lambda: defaultdict(float))
            for s in set(list(Q1.keys()) + list(Q2.keys())):
                for a in ACTIONS:
                    Q_combined[s][a] = (Q1[s][a] + Q2[s][a]) / 2
            
            # Use combined Q for action selection
            action = epsilon_greedy(Q_combined, state, epsilon)
            next_state = env.get_next_state(state, action)
            reward = env.get_reward(state, next_state)
            
            # Randomly update either Q1 or Q2
            if random.random() < 0.5:
                # Update Q1
                if env.is_terminal(next_state):
                    Q1[state][action] += alpha * (reward - Q1[state][action])
                else:
                    # Use Q1 to determine the best action, but Q2 to estimate its value
                    best_action = max(ACTIONS, key=lambda a: Q1[next_state][a])
                    Q1[state][action] += alpha * (reward + gamma * Q2[next_state][best_action] - Q1[state][action])
            else:
                # Update Q2
                if env.is_terminal(next_state):
                    Q2[state][action] += alpha * (reward - Q2[state][action])
                else:
                    # Use Q2 to determine the best action, but Q1 to estimate its value
                    best_action = max(ACTIONS, key=lambda a: Q2[next_state][a])
                    Q2[state][action] += alpha * (reward + gamma * Q1[next_state][best_action] - Q2[state][action])
            
            state = next_state
            steps += 1
            
            # Prevent infinite loops
            if steps > 1000:
                break
        
        steps_per_episode.append(steps)
        
        # Evaluate policy periodically
        if (episode + 1) % eval_interval == 0:
            # Create a combined Q-function for evaluation
            Q_combined = defaultdict(lambda: defaultdict(float))
            for s in set(list(Q1.keys()) + list(Q2.keys())):
                for a in ACTIONS:
                    Q_combined[s][a] = (Q1[s][a] + Q2[s][a]) / 2
            
            eval_steps.append(evaluate_policy(env, Q_combined, num_episodes=10))
    
    # Create a combined Q-function for return
    Q_combined = defaultdict(lambda: defaultdict(float))
    for s in set(list(Q1.keys()) + list(Q2.keys())):
        for a in ACTIONS:
            Q_combined[s][a] = (Q1[s][a] + Q2[s][a]) / 2
    
    return Q_combined, steps_per_episode, eval_steps


def multi_agent_q_learning(env, epsilon, alpha, gamma=1.0, max_episodes=500):
    """Implement Q-learning for multiple agents"""
    # Initialize Q-values for each agent
    Q_list = [defaultdict(lambda: defaultdict(float)) for _ in range(env.num_agents)]
    steps_per_episode = []
    rewards_per_episode = []
    
    for episode in range(max_episodes):
        states = env.reset()
        total_reward = 0
        steps = 0
        terminal = False
        
        while not terminal and steps < 1000:
            # Select actions for all agents using epsilon-greedy
            actions = [epsilon_greedy(Q, state, epsilon) for Q, state in zip(Q_list, states)]
            
            # Execute actions and observe results
            next_states, rewards, terminal = env.step(states, actions)
            total_reward += sum(rewards)
            
            # Update Q-values for each agent
            for i in range(env.num_agents):
                if not terminal:
                    # Use max Q-value for next state
                    next_max_q = max(Q_list[i][next_states[i]][a] for a in ACTIONS)
                    Q_list[i][states[i]][actions[i]] += alpha * (rewards[i] + gamma * next_max_q - Q_list[i][states[i]][actions[i]])
                else:
                    # Terminal state has no future reward
                    Q_list[i][states[i]][actions[i]] += alpha * (rewards[i] - Q_list[i][states[i]][actions[i]])
            
            states = next_states
            steps += 1
        
        steps_per_episode.append(steps)
        rewards_per_episode.append(total_reward)
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_steps = np.mean(steps_per_episode[-50:])
            avg_reward = np.mean(rewards_per_episode[-50:])
            print(f"Episode {episode+1}/{max_episodes}: "
                  f"Avg Steps={avg_steps:.2f}, "
                  f"Avg Reward={avg_reward:.2f}")
    
    return Q_list, steps_per_episode, rewards_per_episode


#########################
# Evaluation Functions
#########################

def evaluate_policy(env, Q, num_episodes=10, epsilon=0):
    """Evaluate a policy for a given number of episodes"""
    total_steps = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        steps = 0
        
        while not env.is_terminal(state) and steps < 1000:
            action = epsilon_greedy(Q, state, epsilon)
            state = env.get_next_state(state, action)
            steps += 1
        
        total_steps += steps
    
    return total_steps / num_episodes


def detect_convergence(steps_history, window=50, threshold=0.05):
    """Detect when the learning algorithm has converged"""
    if len(steps_history) < window + 1:
        return len(steps_history) - 1
    
    for i in range(len(steps_history) - window):
        window_avg = np.mean(steps_history[i:i+window])
        next_window_avg = np.mean(steps_history[i+1:i+window+1])
        
        # Check if the change in average steps is below threshold
        if abs(window_avg - next_window_avg) / max(1, window_avg) < threshold:
            return i + window
    
    return len(steps_history) - 1


#########################
# Visualization Functions
#########################

def plot_learning_curves(results, algorithm, grid_type, epsilons, alphas, smoothing=5):
    """Plot learning curves for various parameter combinations"""
    plt.figure(figsize=(12, 8))
    
    for epsilon in epsilons:
        for alpha in alphas:
            key = (algorithm, grid_type, epsilon, alpha)
            if key in results:
                steps = results[key]['steps']
                
                # Apply smoothing
                if smoothing > 1:
                    steps = np.array(steps)
                    smoothed = np.array([np.mean(steps[max(0, i-smoothing):i+1]) 
                                        for i in range(len(steps))])
                    plt.plot(smoothed, label=f"ε={epsilon}, α={alpha}")
                else:
                    plt.plot(steps, label=f"ε={epsilon}, α={alpha}")
    
    plt.xlabel("Episodes")
    plt.ylabel("Steps to Goal")
    plt.title(f"Learning Curve for {algorithm.upper()} on Gridworld {grid_type}")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(f"plots/{algorithm}_gridworld_{grid_type}.png")
    plt.close()


def plot_algorithm_comparison(results, grid_type, epsilon, alpha):
    """Compare learning curves for different algorithms with the same parameters"""
    plt.figure(figsize=(12, 8))
    
    algorithms = ['sarsa', 'q_learning', 'double_q_learning']
    colors = ['blue', 'red', 'green']
    
    for algo, color in zip(algorithms, colors):
        key = (algo, grid_type, epsilon, alpha)
        if key in results:
            steps = results[key]['steps']
            smoothed = np.array([np.mean(steps[max(0, i-10):i+1]) 
                                for i in range(len(steps))])
            plt.plot(smoothed, label=algo.upper(), color=color)
    
    plt.xlabel("Episodes")
    plt.ylabel("Steps to Goal")
    plt.title(f"Algorithm Comparison on Gridworld {grid_type} (ε={epsilon}, α={alpha})")
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(f"plots/comparison_gridworld_{grid_type}_e{epsilon}_a{alpha}.png")
    plt.close()


def plot_multi_agent_results(steps, rewards, epsilon, alpha):
    """Plot results from multi-agent learning"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Plot steps per episode
    smoothed_steps = np.array([np.mean(steps[max(0, i-10):i+1]) 
                              for i in range(len(steps))])
    ax1.plot(smoothed_steps)
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Steps to Complete Episode")
    ax1.set_title(f"Multi-Agent Learning Curve (ε={epsilon}, α={alpha})")
    ax1.grid(True)
    
    # Plot rewards per episode
    smoothed_rewards = np.array([np.mean(rewards[max(0, i-10):i+1]) 
                                for i in range(len(rewards))])
    ax2.plot(smoothed_rewards)
    ax2.set_xlabel("Episodes")
    ax2.set_ylabel("Total Reward")
    ax2.set_title(f"Multi-Agent Reward Curve (ε={epsilon}, α={alpha})")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"plots/multi_agent_e{epsilon}_a{alpha}.png")
    plt.close()


#########################
# Experiment Runners
#########################

def run_task1_experiments():
    """Run experiments for Task 1"""
    print("Starting Task 1 experiments...")
    
    # Parameters to test
    algorithms = ['sarsa', 'q_learning', 'double_q_learning']
    grid_types = ['A', 'B']
    epsilons = [0.1, 0.2, 0.3]
    alphas = [0.1, 0.3, 0.5]
    max_episodes = 500
    num_runs = 5  # Run each experiment multiple times for statistical significance
    
    # Store results
    all_results = {}
    metrics = {}
    
    for grid_type in grid_types:
        print(f"\nRunning experiments for Gridworld {grid_type}")
        env = GridWorld(grid_type)
        
        for algorithm in algorithms:
            print(f"  Algorithm: {algorithm}")
            
            for epsilon in epsilons:
                for alpha in alphas:
                    print(f"    Parameters: ε={epsilon}, α={alpha}")
                    
                    # Run multiple times to ensure statistical significance
                    all_steps = []
                    all_eval_steps = []
                    
                    for run in range(num_runs):
                        if algorithm == 'sarsa':
                            Q, steps, eval_steps = sarsa(env, epsilon, alpha, 
                                                        max_episodes=max_episodes)
                        elif algorithm == 'q_learning':
                            Q, steps, eval_steps = q_learning(env, epsilon, alpha, 
                                                             max_episodes=max_episodes)
                        elif algorithm == 'double_q_learning':
                            Q, steps, eval_steps = double_q_learning(env, epsilon, alpha, 
                                                                    max_episodes=max_episodes)
                        
                        all_steps.append(steps)
                        all_eval_steps.append(eval_steps)
                    
                    # Average results across runs
                    avg_steps = np.mean(all_steps, axis=0).tolist()
                    avg_eval_steps = np.mean(all_eval_steps, axis=0).tolist()
                    
                    # Detect convergence
                    convergence_episode = detect_convergence(avg_steps)
                    
                    # Store results - but convert Q to a regular dict
                    key = (algorithm, grid_type, epsilon, alpha)
                    all_results[key] = {
                        'steps': avg_steps,
                        'eval_steps': avg_eval_steps,
                        # Don't store Q-functions for saving - they're not picklable
                    }
                    
                    # Calculate metrics
                    metrics[key] = {
                        'convergence_episode': convergence_episode,
                        'avg_final_steps': np.mean(avg_steps[-50:]),
                        'std_final_steps': np.std(avg_steps[-50:])
                    }
    
    # Plot learning curves
    for algorithm in algorithms:
        for grid_type in grid_types:
            plot_learning_curves(all_results, algorithm, grid_type, epsilons, alphas)
    
    # Plot algorithm comparisons with best parameters
    for grid_type in grid_types:
        for epsilon in [0.1]:  # Choose specific epsilon for comparison
            for alpha in [0.3]:  # Choose specific alpha for comparison
                plot_algorithm_comparison(all_results, grid_type, epsilon, alpha)
    
    # Analyze results
    analyze_results(algorithms, grid_types, epsilons, alphas, [max_episodes], metrics)
    
    # Save results to file (without Q functions)
    try:
        import pickle
        with open('results/task1_results.pkl', 'wb') as f:
            pickle.dump({
                'steps_results': {k: v['steps'] for k, v in all_results.items()},
                'eval_steps_results': {k: v['eval_steps'] for k, v in all_results.items()},
                'metrics': metrics
            }, f)
        print("Results saved to results/task1_results.pkl")
    except Exception as e:
        print(f"Warning: Could not save results: {e}")
    
    print("Task 1 experiments completed and results saved.")
    return all_results, metrics


def run_task2_experiments():
    """Run experiments for Task 2"""
    print("\nStarting Task 2 experiments (Multi-Agent)...")
    
    # Parameters to test
    epsilons = [0.1, 0.2, 0.3]
    alphas = [0.1, 0.3, 0.5]
    max_episodes = 500
    num_runs = 3
    
    # Store results
    multi_agent_results = {}
    
    env = MultiAgentGridWorld(num_agents=3)
    
    for epsilon in epsilons:
        for alpha in alphas:
            print(f"  Parameters: ε={epsilon}, α={alpha}")
            
            all_steps = []
            all_rewards = []
            
            for run in range(num_runs):
                print(f"    Run {run+1}/{num_runs}")
                Q_list, steps, rewards = multi_agent_q_learning(
                    env, epsilon, alpha, max_episodes=max_episodes)
                
                all_steps.append(steps)
                all_rewards.append(rewards)
            
            # Average results across runs
            avg_steps = np.mean(all_steps, axis=0).tolist()
            avg_rewards = np.mean(all_rewards, axis=0).tolist()
            
            # Store results (without Q_list)
            key = (epsilon, alpha)
            multi_agent_results[key] = {
                'steps': avg_steps,
                'rewards': avg_rewards,
                # Don't store Q_list for saving
            }
            
            # Plot results
            plot_multi_agent_results(avg_steps, avg_rewards, epsilon, alpha)
    
    # Save results to file (without Q_list)
    try:
        import pickle
        with open('results/task2_results.pkl', 'wb') as f:
            pickle.dump(multi_agent_results, f)
        print("Results saved to results/task2_results.pkl")
    except Exception as e:
        print(f"Warning: Could not save results: {e}")
    
    print("Task 2 experiments completed and results saved.")
    return multi_agent_results


# Fix for matplotlib QSocketNotifier warning
def main():
    """Main function to run all experiments"""
    print("MAS Written Assignment - HW No.1")
    print("Reinforcement Learning in Grid Worlds")
    print("====================================")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Configure matplotlib to use non-interactive backend
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    
    # Run experiments for Task 1
    task1_results, task1_metrics = run_task1_experiments()
    
    # Run experiments for Task 2
    task2_results = run_task2_experiments()
    
    print("\nAll experiments completed successfully!")
    print("Results are saved in the 'results' directory")
    print("Plots are saved in the 'plots' directory")
    print("\nYou can now analyze the results and write your report.")

if __name__ == "__main__":
    main()