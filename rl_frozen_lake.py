import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt


N_STATES   = 16        
N_ACTIONS  = 4
EPISODES   = 5000
ALPHA      = 0.5       
GAMMA      = 0.99      
EPSILON    = 1.0       
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995


def epsilon_greedy(Q, state, epsilon):

    if np.random.random() < epsilon:
        return np.random.randint(N_ACTIONS)   
    return np.argmax(Q[state])                 


def moving_average(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode='valid')


def q_learning(episodes=EPISODES):

    env = gym.make("FrozenLake-v1", is_slippery=False)
    Q = np.zeros((N_STATES, N_ACTIONS))
    epsilon = EPSILON
    rewards_per_episode = []
    start = time.time()

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-Learning update (uses max over next actions)
            best_next = np.max(Q[next_state])
            Q[state, action] += ALPHA * (reward + GAMMA * best_next - Q[state, action])

            state = next_state
            total_reward += reward

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        rewards_per_episode.append(total_reward)

    elapsed = time.time() - start
    env.close()
    return Q, rewards_per_episode, elapsed


def sarsa(episodes=EPISODES):
    env = gym.make("FrozenLake-v1", is_slippery=False)
    Q = np.zeros((N_STATES, N_ACTIONS))
    epsilon = EPSILON
    sarsa_decay = 0.999   
    rewards_per_episode = []
    start = time.time()

    for ep in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        total_reward = 0
        done = False

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            next_action = epsilon_greedy(Q, next_state, epsilon)

            Q[state, action] += ALPHA * (reward + GAMMA * Q[next_state, next_action] - Q[state, action])

            state = next_state
            action = next_action
            total_reward += reward

        epsilon = max(EPSILON_MIN, 1.0 - ep / (episodes * 0.8))
        rewards_per_episode.append(total_reward)

    elapsed = time.time() - start
    env.close()
    return Q, rewards_per_episode, elapsed


def monte_carlo(episodes=EPISODES):
    env = gym.make("FrozenLake-v1", is_slippery=False)
    Q = np.zeros((N_STATES, N_ACTIONS))
    returns_sum   = np.zeros((N_STATES, N_ACTIONS))
    returns_count = np.zeros((N_STATES, N_ACTIONS))
    rewards_per_episode = []
    start = time.time()
    MAX_STEPS = 200

    for ep in range(episodes):
        epsilon = max(0.05, 1.0 - ep / (episodes * 0.7))

        state, _ = env.reset()
        episode_data = []
        done = False
        steps = 0

        while not done and steps < MAX_STEPS:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_data.append((state, action, reward))
            state = next_state
            steps += 1

        total_reward = sum(r for _, _, r in episode_data)
        rewards_per_episode.append(total_reward)

        G = 0
        for s, a, r in reversed(episode_data):
            G = r + GAMMA * G
            returns_sum[s, a]   += G
            returns_count[s, a] += 1
            Q[s, a] = returns_sum[s, a] / returns_count[s, a]

    elapsed = time.time() - start
    env.close()
    return Q, rewards_per_episode, elapsed


def evaluate_policy(Q, n_eval=100):
    env = gym.make("FrozenLake-v1", is_slippery=False)
    wins = 0
    for _ in range(n_eval):
        state, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < 100:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        if reward == 1.0:
            wins += 1
    env.close()
    return wins / n_eval * 100  


def plot_results(ql_rewards, sarsa_rewards, mc_rewards):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RL Algorithms on FrozenLake-v1 (4×4, non-slippery)", fontsize=14)

    ax = axes[0]
    window = 200
    ax.plot(moving_average(ql_rewards,    window), label="Q-Learning",   color="#1f77b4")
    ax.plot(moving_average(sarsa_rewards, window), label="SARSA",        color="#ff7f0e")
    ax.plot(moving_average(mc_rewards,    window), label="Monte Carlo",   color="#2ca02c")
    ax.set_title(f"Learning Curves (smoothed, window={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for rewards, label, color in [
        (ql_rewards, "Q-Learning", "#1f77b4"),
        (sarsa_rewards, "SARSA",   "#ff7f0e"),
        (mc_rewards, "Monte Carlo","#2ca02c"),
    ]:
        cumulative = np.cumsum(rewards)
        ax.plot(cumulative, label=label, color=color)
    ax.set_title("Cumulative Wins Over Training")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Wins")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rl_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved as rl_results.png")


def print_policy(Q, name):
    arrows = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    holes  = {5, 7, 11, 12}   
    goal   = 15
    print(f"\n{'─'*30}")
    print(f"  Optimal Policy: {name}")
    print(f"{'─'*30}")
    for row in range(4):
        line = ""
        for col in range(4):
            s = row * 4 + col
            if s in holes:
                line += " H "
            elif s == goal:
                line += " G "
            elif s == 0:
                line += " S "
            else:
                line += f" {arrows[np.argmax(Q[s])]} "
        print(line)


if __name__ == "__main__":
    print("=" * 50)
    print("  RL Final Project — FrozenLake-v1")
    print("=" * 50)
    print(f"  Episodes: {EPISODES} | α={ALPHA} | γ={GAMMA}")
    print()

    print("▶ Training Q-Learning...")
    ql_Q, ql_r, ql_t   = q_learning()

    print("▶ Training SARSA...")
    sarsa_Q, sarsa_r, sarsa_t = sarsa()

    print("▶ Training Monte Carlo...")
    mc_Q, mc_r, mc_t    = monte_carlo()

    ql_win    = evaluate_policy(ql_Q)
    sarsa_win = evaluate_policy(sarsa_Q)
    mc_win    = evaluate_policy(mc_Q)

    print("\n" + "=" * 50)
    print("  RESULTS COMPARISON")
    print("=" * 50)
    print(f"{'Algorithm':<15} {'Train Time':>12} {'Win Rate':>10}")
    print("-" * 40)
    print(f"{'Q-Learning':<15} {ql_t:>10.2f}s  {ql_win:>8.1f}%")
    print(f"{'SARSA':<15} {sarsa_t:>10.2f}s  {sarsa_win:>8.1f}%")
    print(f"{'Monte Carlo':<15} {mc_t:>10.2f}s  {mc_win:>8.1f}%")
    print("=" * 50)

    print_policy(ql_Q,    "Q-Learning")
    print_policy(sarsa_Q, "SARSA")
    print_policy(mc_Q,    "Monte Carlo")

    plot_results(ql_r, sarsa_r, mc_r)
