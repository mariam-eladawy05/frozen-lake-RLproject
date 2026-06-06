import pygame
import numpy as np
import time
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID_SIZE = 8
CELL_SIZE = 80
SIDE_PANEL_WIDTH = 300
TOP_BAR_HEIGHT = 50

WINDOW_W = GRID_SIZE * CELL_SIZE + SIDE_PANEL_WIDTH
WINDOW_H = GRID_SIZE * CELL_SIZE + TOP_BAR_HEIGHT

WHITE = (255, 255, 255)
BLACK = (20, 20, 20)
GRAY = (200, 200, 200)
LIGHT_GRAY = (240, 240, 240)
AGENT_COL = (30, 120, 255)
GOAL_COL = (50, 200, 80)
OBS_COL = (220, 60, 60)
TRAIL_COL = (180, 210, 255)
PANEL_COL = (245, 245, 248)
TEXT_COL = (40, 40, 40)
Q_COLOR = (30, 120, 255)
S_COLOR = (255, 140, 0)

GRID_MAP = [
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 2],
]

START_POS = (0, 0)
GOAL_POS = (7, 7)

N_STATES = GRID_SIZE * GRID_SIZE
N_ACTIONS = 4

ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}

ACTION_NAMES = {
    0: "Up",
    1: "Down",
    2: "Left",
    3: "Right",
}

ALPHA = 0.3
GAMMA = 0.99
EPISODES = 3000
MAX_STEPS = 300

EPSILON_START = 1.0
EPSILON_MIN = 0.01

def pos_to_state(row, col):
    return row * GRID_SIZE + col


def state_to_pos(state):
    return divmod(state, GRID_SIZE)


def reset():
    return pos_to_state(*START_POS)


def is_valid(row, col):
    if row < 0 or row >= GRID_SIZE:
        return False
    if col < 0 or col >= GRID_SIZE:
        return False
    return GRID_MAP[row][col] != 1


def step(state, action):
    row, col = state_to_pos(state)
    dr, dc = ACTION_DELTAS[action]

    next_row = row + dr
    next_col = col + dc

    if not is_valid(next_row, next_col):
        return state, -0.5, False

    next_state = pos_to_state(next_row, next_col)

    if (next_row, next_col) == GOAL_POS:
        return next_state, 10.0, True

    return next_state, -0.1, False


def epsilon_greedy(Q, state, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(N_ACTIONS)
    return int(np.argmax(Q[state]))


def epsilon_decay(ep):
    return max(EPSILON_MIN, EPSILON_START - ep / (EPISODES * 0.8))


def train_q_learning():
    Q = np.zeros((N_STATES, N_ACTIONS))

    rewards_log = []
    steps_log = []

    start_time = time.time()

    for ep in range(EPISODES):
        state = reset()
        epsilon = epsilon_decay(ep)

        total_reward = 0
        steps = 0
        done = False

        while not done and steps < MAX_STEPS:
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done = step(state, action)

            if done:
                target = reward
            else:
                target = reward + GAMMA * np.max(Q[next_state])

            Q[state, action] += ALPHA * (target - Q[state, action])

            state = next_state
            total_reward += reward
            steps += 1

        rewards_log.append(total_reward)
        steps_log.append(steps)

    elapsed = time.time() - start_time
    return Q, rewards_log, steps_log, elapsed



def train_sarsa():
    Q = np.zeros((N_STATES, N_ACTIONS))

    rewards_log = []
    steps_log = []

    start_time = time.time()

    for ep in range(EPISODES):
        state = reset()
        epsilon = epsilon_decay(ep)
        action = epsilon_greedy(Q, state, epsilon)

        total_reward = 0
        steps = 0
        done = False

        while not done and steps < MAX_STEPS:
            next_state, reward, done = step(state, action)

            # Correct terminal-state SARSA handling
            if done:
                target = reward
            else:
                next_action = epsilon_greedy(Q, next_state, epsilon)
                target = reward + GAMMA * Q[next_state, next_action]

            Q[state, action] += ALPHA * (target - Q[state, action])

            state = next_state

            if not done:
                action = next_action

            total_reward += reward
            steps += 1

        rewards_log.append(total_reward)
        steps_log.append(steps)

    elapsed = time.time() - start_time
    return Q, rewards_log, steps_log, elapsed



def get_greedy_path(Q):
    state = reset()
    path = [state_to_pos(state)]

    visited = set()
    visited.add(state)

    done = False
    steps = 0

    while not done and steps < 100:
        action = int(np.argmax(Q[state]))
        next_state, reward, done = step(state, action)

        path.append(state_to_pos(next_state))

        if next_state in visited and not done:
            break

        visited.add(next_state)
        state = next_state
        steps += 1

    return path


def eval_win_rate(Q, n=200):
    wins = 0
    total_steps = 0

    for _ in range(n):
        state = reset()
        done = False
        steps = 0

        while not done and steps < 200:
            action = int(np.argmax(Q[state]))
            state, reward, done = step(state, action)
            steps += 1

        if done and state_to_pos(state) == GOAL_POS:
            wins += 1
            total_steps += steps

    win_rate = wins / n * 100
    avg_steps = total_steps / wins if wins > 0 else 0

    return win_rate, avg_steps



def smooth(data, window=100):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def save_plot(ql_rewards, sarsa_rewards, ql_steps, sarsa_steps):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Robot Navigation — Q-Learning vs SARSA", fontsize=13)

    ax = axes[0]
    ax.plot(smooth(ql_rewards), label="Q-Learning")
    ax.plot(smooth(sarsa_rewards), label="SARSA")
    ax.set_title("Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(smooth(ql_steps), label="Q-Learning")
    ax.plot(smooth(sarsa_steps), label="SARSA")
    ax.set_title("Steps per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("simulation_results.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Plot saved as simulation_results.png")


def draw_grid(screen, path=None, agent_pos=None, label="", color=AGENT_COL):
    screen.fill(LIGHT_GRAY)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x = col * CELL_SIZE
            y = row * CELL_SIZE + TOP_BAR_HEIGHT

            rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
            cell = GRID_MAP[row][col]

            if cell == 1:
                pygame.draw.rect(screen, OBS_COL, rect)
                pygame.draw.line(screen, (180, 40, 40), (x, y), (x + CELL_SIZE, y + CELL_SIZE), 2)
                pygame.draw.line(screen, (180, 40, 40), (x + CELL_SIZE, y), (x, y + CELL_SIZE), 2)

            elif cell == 2:
                pygame.draw.rect(screen, GOAL_COL, rect)

            else:
                pygame.draw.rect(screen, WHITE, rect)

            pygame.draw.rect(screen, GRAY, rect, 1)

    if path:
        for r, c in path[:-1]:
            x = c * CELL_SIZE + CELL_SIZE // 2
            y = r * CELL_SIZE + CELL_SIZE // 2 + TOP_BAR_HEIGHT
            pygame.draw.circle(screen, TRAIL_COL, (x, y), 8)

    if agent_pos:
        r, c = agent_pos
        x = c * CELL_SIZE + CELL_SIZE // 2
        y = r * CELL_SIZE + CELL_SIZE // 2 + TOP_BAR_HEIGHT

        pygame.draw.circle(screen, color, (x, y), 22)
        pygame.draw.circle(screen, WHITE, (x, y), 14)
        pygame.draw.circle(screen, color, (x, y), 8)

        for dx, dy in [(-18, -10), (18, -10), (-18, 10), (18, 10)]:
            pygame.draw.circle(screen, BLACK, (x + dx, y + dy), 5)

    goal_x = GOAL_POS[1] * CELL_SIZE + CELL_SIZE // 2
    goal_y = GOAL_POS[0] * CELL_SIZE + CELL_SIZE // 2 + TOP_BAR_HEIGHT

    font_sm = pygame.font.SysFont("Arial", 18, bold=True)
    goal_text = font_sm.render("GOAL", True, WHITE)
    screen.blit(goal_text, goal_text.get_rect(center=(goal_x, goal_y)))

    pygame.draw.rect(screen, (30, 70, 140), (0, 0, GRID_SIZE * CELL_SIZE, TOP_BAR_HEIGHT))

    font_title = pygame.font.SysFont("Arial", 20, bold=True)
    title = font_title.render(f"Robot Navigation Simulator — {label}", True, WHITE)
    screen.blit(title, (10, 12))


def draw_panel(screen, algo_name, total_eps, reward, steps, elapsed, win_rate, avg_steps, color):
    px = GRID_SIZE * CELL_SIZE + 10
    pw = SIDE_PANEL_WIDTH - 20
    ph = WINDOW_H

    pygame.draw.rect(screen, PANEL_COL, (px, 0, pw, ph))
    pygame.draw.line(screen, GRAY, (px, 0), (px, ph), 2)

    font_h = pygame.font.SysFont("Arial", 16, bold=True)
    font_b = pygame.font.SysFont("Arial", 14)
    font_s = pygame.font.SysFont("Arial", 13)

    y = 20

    badge_rect = pygame.Rect(px + 10, y, pw - 20, 32)
    pygame.draw.rect(screen, color, badge_rect, border_radius=6)

    badge_text = font_h.render(algo_name, True, WHITE)
    screen.blit(badge_text, badge_text.get_rect(center=badge_rect.center))

    y += 55

    def stat_row(label, value, highlight=False):
        nonlocal y
        text_color = color if highlight else TEXT_COL

        label_text = font_s.render(label, True, GRAY)
        value_text = font_b.render(str(value), True, text_color)

        screen.blit(label_text, (px + 12, y))
        screen.blit(value_text, (px + pw - value_text.get_width() - 12, y))

        y += 25

    stat_row("Episodes trained:", f"{total_eps:,}")
    stat_row("Path steps:", f"{steps}")
    stat_row("Path reward:", f"{reward:.1f}", True)
    stat_row("Train time:", f"{elapsed:.2f}s")
    stat_row("Win rate:", f"{win_rate:.0f}%", True)
    stat_row("Avg eval steps:", f"{avg_steps:.1f}")

    y += 15
    pygame.draw.rect(screen, GRAY, (px + 10, y, pw - 20, 1))
    y += 15

    screen.blit(font_s.render("Legend", True, TEXT_COL), (px + 12, y))
    y += 25

    legend_items = [
        (AGENT_COL, "Agent / robot"),
        (GOAL_COL, "Goal"),
        (OBS_COL, "Obstacle"),
        (TRAIL_COL, "Path trail"),
    ]

    for col, label in legend_items:
        pygame.draw.circle(screen, col, (px + 22, y + 7), 7)
        screen.blit(font_s.render(label, True, TEXT_COL), (px + 38, y))
        y += 23

    y += 15
    pygame.draw.rect(screen, GRAY, (px + 10, y, pw - 20, 1))
    y += 15

    screen.blit(font_s.render("Controls", True, TEXT_COL), (px + 12, y))
    y += 25

    screen.blit(font_s.render("Q: Q-Learning demo", True, TEXT_COL), (px + 12, y))
    y += 22

    screen.blit(font_s.render("S: SARSA demo", True, TEXT_COL), (px + 12, y))
    y += 22

    screen.blit(font_s.render("ESC: Quit", True, TEXT_COL), (px + 12, y))


def calculate_path_reward(path):
    reward = 0

    for i in range(1, len(path)):
        if path[i] == GOAL_POS:
            reward += 10.0
        else:
            reward -= 0.1

    return reward


def animate_path(screen, path, algo_name, elapsed, win_rate, avg_steps, color, clock):
    path_steps = len(path) - 1
    path_reward = calculate_path_reward(path)

    for i, pos in enumerate(path):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

        draw_grid(
            screen,
            path=path[: i + 1],
            agent_pos=pos,
            label=algo_name,
            color=color,
        )

        draw_panel(
            screen,
            algo_name=algo_name,
            total_eps=EPISODES,
            reward=path_reward,
            steps=path_steps,
            elapsed=elapsed,
            win_rate=win_rate,
            avg_steps=avg_steps,
            color=color,
        )

        pygame.display.flip()
        clock.tick(4)

    time.sleep(1.2)


def main():
    print("=" * 55)
    print("RL Simulation — Robot Navigation Using PyGame")
    print("=" * 55)
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE}")
    print(f"States: {N_STATES}")
    print(f"Actions: {N_ACTIONS}")
    print(f"Episodes: {EPISODES}")
    print("Episode ends when the agent reaches the goal or hits max steps.")
    print("Hitting an obstacle gives -0.5 penalty but does not end the episode.")
    print()

    print("Training Q-Learning...")
    ql_Q, ql_rewards, ql_steps, ql_time = train_q_learning()
    ql_win, ql_avg_steps = eval_win_rate(ql_Q)
    ql_path = get_greedy_path(ql_Q)

    print(f"Q-Learning done in {ql_time:.2f}s")
    print(f"Win rate: {ql_win:.0f}%")
    print(f"Average evaluation steps: {ql_avg_steps:.1f}")
    print(f"Greedy path length: {len(ql_path) - 1} steps")
    print()

    print("Training SARSA...")
    sarsa_Q, sarsa_rewards, sarsa_steps, sarsa_time = train_sarsa()
    sarsa_win, sarsa_avg_steps = eval_win_rate(sarsa_Q)
    sarsa_path = get_greedy_path(sarsa_Q)

    print(f"SARSA done in {sarsa_time:.2f}s")
    print(f"Win rate: {sarsa_win:.0f}%")
    print(f"Average evaluation steps: {sarsa_avg_steps:.1f}")
    print(f"Greedy path length: {len(sarsa_path) - 1} steps")
    print()

    save_plot(ql_rewards, sarsa_rewards, ql_steps, sarsa_steps)

    pygame.init()

    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("RL Robot Navigation Simulator")

    clock = pygame.time.Clock()

    draw_grid(screen, label="Ready — Press Q or S")
    draw_panel(
        screen,
        algo_name="Choose Algorithm",
        total_eps=EPISODES,
        reward=0,
        steps=0,
        elapsed=0,
        win_rate=0,
        avg_steps=0,
        color=AGENT_COL,
    )

    pygame.display.flip()

    print("PyGame window is open.")
    print("Press Q for Q-Learning demo.")
    print("Press S for SARSA demo.")
    print("Press ESC to quit.")

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_q:
                    print("Animating Q-Learning path...")
                    animate_path(
                        screen,
                        path=ql_path,
                        algo_name="Q-Learning",
                        elapsed=ql_time,
                        win_rate=ql_win,
                        avg_steps=ql_avg_steps,
                        color=Q_COLOR,
                        clock=clock,
                    )

                    draw_grid(screen, label="Press Q or S to replay")
                    draw_panel(
                        screen,
                        algo_name="Choose Algorithm",
                        total_eps=EPISODES,
                        reward=0,
                        steps=0,
                        elapsed=0,
                        win_rate=0,
                        avg_steps=0,
                        color=AGENT_COL,
                    )
                    pygame.display.flip()

                elif event.key == pygame.K_s:
                    print("Animating SARSA path...")
                    animate_path(
                        screen,
                        path=sarsa_path,
                        algo_name="SARSA",
                        elapsed=sarsa_time,
                        win_rate=sarsa_win,
                        avg_steps=sarsa_avg_steps,
                        color=S_COLOR,
                        clock=clock,
                    )

                    draw_grid(screen, label="Press Q or S to replay")
                    draw_panel(
                        screen,
                        algo_name="Choose Algorithm",
                        total_eps=EPISODES,
                        reward=0,
                        steps=0,
                        elapsed=0,
                        win_rate=0,
                        avg_steps=0,
                        color=AGENT_COL,
                    )
                    pygame.display.flip()

        clock.tick(30)

    pygame.quit()
    print("Simulation ended.")


if __name__ == "__main__":
    main()