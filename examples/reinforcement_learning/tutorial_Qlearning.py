"""
Q-Table learning algorithm for FrozenLake-v1 (Gymnasium)
Non deep learning - TD Learning, Off-Policy, e-Greedy Exploration
Q(S, A) <- Q(S, A) + alpha * (R + lambda * Q(newS, newA) - Q(S, A))
"""

import argparse
import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

# ----------------- Arguments ----------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)
parser.add_argument('--seed', help='random seed', type=int, default=0)
parser.add_argument('--env_id', default='FrozenLake-v1')
args = parser.parse_args()

# ----------------- Environment ----------------- #
alg_name = 'Qlearning'
env_id = args.env_id
env = gym.make(env_id)
render = False  # display the game environment

np.random.seed(args.seed)

# ----------------- Q-Table Initialization ----------------- #
Q = np.zeros([env.observation_space.n, env.action_space.n])
lr = 0.85       # learning rate (alpha)
lambd = 0.99    # discount factor
num_episodes = 10
t0 = time.time()

# ----------------- Training ----------------- #
if args.train:
    all_episode_reward = []
    for i in range(num_episodes):
        s, _ = env.reset()        # reset returns (state, info)
        s = int(s)                # ensure integer index
        rAll = 0
        for j in range(99):
            if render:
                env.render()
            # e-greedy action selection with noise
            a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
            s1, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s1 = int(s1)
            # Q-Learning update
            Q[s, a] = Q[s, a] + lr * (r + lambd * np.max(Q[s1, :]) - Q[s, a])
            rAll += r
            s = s1
            if done:
                break
        # running average for plotting
        if i == 0:
            all_episode_reward.append(rAll)
        else:
            all_episode_reward.append(all_episode_reward[-1] * 0.9 + rAll * 0.1)

        print(f"Training  | Episode: {i+1}/{num_episodes}  | Episode Reward: {rAll:.4f}  | Running Time: {time.time() - t0:.4f}")

    # Save Q-Table
    path = os.path.join('model', f"{alg_name}_{env_id}")
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, 'Q_table.npy'), Q)

    # Plot rewards
    os.makedirs('image', exist_ok=True)
    plt.plot(all_episode_reward)
    plt.xlabel('Episode')
    plt.ylabel('Running Average Reward')
    plt.title('Training Progress')
    plt.savefig(os.path.join('image', f"{alg_name}_{env_id}.png"))

# ----------------- Testing ----------------- #
if args.test:
    path = os.path.join('model', f"{alg_name}_{env_id}")
    Q = np.load(os.path.join(path, 'Q_table.npy'))

    for i in range(100):
        s, _ = env.reset()
        s = int(s)
        rAll = 0
        for j in range(99):
            a = np.argmax(Q[s, :])
            s1, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s1 = int(s1)
            rAll += r
            s = s1
            if done:
                break
        print(f"Testing  | Episode: {i+1}/100  | Episode Reward: {rAll:.4f}  | Running Time: {time.time() - t0:.4f}")
