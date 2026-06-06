# Reinforcement Learning Final Project

A comprehensive Reinforcement Learning project that implements and compares three fundamental RL algorithms—**Q-Learning**, **SARSA**, and **Monte Carlo Control**—on the **FrozenLake-v1** environment from Gymnasium. The project also includes a custom **PyGame Robot Navigation Simulator** where Q-Learning and SARSA are applied to autonomous path planning in a grid-world environment.

## Team Members

* Mariam Eladawy
* Jaslim Mohamed

---

## Project Overview

This project explores how different reinforcement learning algorithms learn optimal policies through interaction with an environment.

The project is divided into two parts:

### Part 1 — FrozenLake-v1

Implementation and comparison of:

* Q-Learning
* SARSA
* Monte Carlo Control

The goal is to train an agent to navigate a 4×4 FrozenLake grid from a start state to a goal state while avoiding holes.

### Part 2 — Robot Navigation Simulator (Bonus)

A custom PyGame simulation where a robot learns to navigate an 8×8 grid environment containing obstacles and a goal location.

Algorithms used:

* Q-Learning
* SARSA

---

## Algorithms Implemented

### Q-Learning

* Off-policy Temporal Difference learning
* Uses the maximum estimated future reward for updates
* Fastest convergence among tested algorithms

### SARSA

* On-policy Temporal Difference learning
* Updates based on the action actually taken
* More conservative and safer near obstacles

### Monte Carlo Control

* Learns from complete episodes
* Uses return averaging instead of bootstrapping
* Unbiased but higher variance

---

## FrozenLake Environment

### Environment Details

* Environment: FrozenLake-v1
* Grid Size: 4×4
* States: 16
* Actions:

  * Left
  * Down
  * Right
  * Up
* Reward:

  * +1 for reaching the goal
  * 0 otherwise
* Deterministic transitions (`is_slippery=False`)

### Training Configuration

| Parameter           | Value       |
| ------------------- | ----------- |
| Learning Rate (α)   | 0.5         |
| Discount Factor (γ) | 0.99        |
| Episodes            | 5000        |
| Initial Epsilon     | 1.0         |
| Minimum Epsilon     | 0.01 / 0.05 |

---

## FrozenLake Results

| Algorithm   | Win Rate | Convergence Episode |
| ----------- | -------- | ------------------- |
| Q-Learning  | 100%     | ~500                |
| SARSA       | 100%     | ~3500               |
| Monte Carlo | 100%     | ~3500               |

### Key Findings

* Q-Learning converged significantly faster.
* SARSA required more exploration but achieved identical final performance.
* Monte Carlo suffered from sparse rewards and slower learning.
* All algorithms ultimately reached a 100% success rate.

---

## Robot Navigation Simulator

### Environment

The simulator consists of:

* 8×8 Grid World
* Robot Agent
* Static Obstacles
* Goal Tile
* Visualized using PyGame

### Rewards

| Event              | Reward |
| ------------------ | ------ |
| Reach Goal         | +10    |
| Valid Move         | -0.1   |
| Collision/Boundary | -0.5   |

### State Representation

```
state = row × 8 + column
```

### Actions

```
0 → Up
1 → Down
2 → Left
3 → Right
```

---

## Robot Navigation Results

| Algorithm  | Win Rate | Avg Path Length | Convergence Episode |
| ---------- | -------- | --------------- | ------------------- |
| Q-Learning | 100%     | ~14 steps       | ~800                |
| SARSA      | 100%     | ~14 steps       | ~1200               |

### Observations

* Q-Learning learns faster and takes more aggressive routes.
* SARSA behaves more cautiously near obstacles.
* Both algorithms successfully discover near-optimal paths.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/reinforcement-learning-final-project.git
cd reinforcement-learning-final-project
```

Install dependencies:

```bash
pip install gymnasium pygame numpy matplotlib
```

---

## Running FrozenLake Experiments

```bash
python q_learning_frozenlake.py
python sarsa_frozenlake.py
python monte_carlo_frozenlake.py
```

---

## Running the Robot Navigation Simulator

```bash
python rl_simulation.py
```

### Controls

| Key | Action               |
| --- | -------------------- |
| Q   | Run Q-Learning Agent |
| S   | Run SARSA Agent      |
| ESC | Exit Simulator       |

---

## Project Structure

```text
├── q_learning_frozenlake.py
├── sarsa_frozenlake.py
├── monte_carlo_frozenlake.py
├── rl_simulation.py
├── rl_results.png
├── simulation_results.png
├── report.pdf
└── README.md
```

---

## Technologies Used

* Python
* Gymnasium (OpenAI Gym)
* NumPy
* Matplotlib
* PyGame

---

## References

* Sutton & Barto, Reinforcement Learning: An Introduction (2nd Edition)
* Gymnasium Documentation
* OpenAI Gym
* Reinforcement Learning Course Materials

---

## Conclusion

This project demonstrates how different reinforcement learning algorithms behave across both benchmark and custom environments. While all algorithms successfully learn optimal policies, Q-Learning consistently converges faster, whereas SARSA provides more conservative behavior. The PyGame simulator further highlights how reinforcement learning techniques can be transferred to practical navigation tasks with minimal changes to the learning framework.
