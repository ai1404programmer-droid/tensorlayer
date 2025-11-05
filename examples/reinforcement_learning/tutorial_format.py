#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Algorithm Name
------------------------
Briefly describe the algorithm, add some details.

Reference
---------
original paper: e.g. https://arxiv.org/pdf/1802.09477.pdf
website: ...

Environment
-----------
OpenAI Gym Pendulum-v0, continuous action space (can be replaced)

Prerequisites
---------------
tensorflow >=2.0.0a0
tensorlayer >=2.0.0
gymnasium >=1.0.0

To run
-------
python tutorial_algorithm.py --train
python tutorial_algorithm.py --test
"""

import argparse
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import gymnasium as gym
import os

# ----------------- reproducible -----------------
np.random.seed(2)
tf.random.set_seed(2)

# ----------------- arguments -----------------
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

# ----------------- hyperparameters -----------------
MAX_EPISODES = 10        # total number of episodes
MAX_STEPS = 10            # maximum steps per episode
GAMMA = 0.99               # discount factor
BATCH_SIZE = 64            # batch size for updates
LR = 1e-3                  # learning rate
HIDDEN_SIZES = [64, 64]    # hidden layers for actor/critic

# ----------------- utility functions -----------------
def plot_rewards(rewards, algo_name='Algorithm', env_name='Env'):
    """Plot episode rewards"""
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.title(f'{algo_name} on {env_name}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(f'plots/{algo_name}_{env_name}.png')
    plt.close()

# ----------------- Algorithm Class Template -----------------
class Algorithm:
    """
    Base class template for RL algorithms
    """

    def __init__(self, state_dim, action_dim, action_bound=None):
        """
        Initialize networks, optimizers, buffers here
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound

        # example: actor network
        self.actor = self.build_actor()
        # example: critic network
        self.critic = self.build_critic()

        # optimizer for actor/critic
        self.actor_optimizer = tf.optimizers.Adam(LR)
        self.critic_optimizer = tf.optimizers.Adam(LR)

    def build_actor(self):
        """Build policy network"""
        inputs = tl.layers.Input([None, self.state_dim], tf.float32)
        x = inputs
        for h in HIDDEN_SIZES:
            x = tl.layers.Dense(h, act=tf.nn.relu)(x)
        # output layer (continuous action example)
        mean = tl.layers.Dense(self.action_dim, act=None)(x)
        model = tl.models.Model(inputs=inputs, outputs=mean)
        model.train()
        return model

    def build_critic(self):
        """Build value network"""
        inputs = tl.layers.Input([None, self.state_dim], tf.float32)
        x = inputs
        for h in HIDDEN_SIZES:
            x = tl.layers.Dense(h, act=tf.nn.relu)(x)
        value = tl.layers.Dense(1, act=None)(x)
        model = tl.models.Model(inputs=inputs, outputs=value)
        model.train()
        return model

    def choose_action(self, state):
        """Return action given state"""
        state = np.array([state], np.float32)
        mean = self.actor(state)
        # example: continuous Gaussian policy
        action = tf.tanh(mean).numpy()[0]
        if self.action_bound is not None:
            action = action * self.action_bound
        return action

    def update(self):
        """
        Update policy and value networks
        (to be implemented in algorithm-specific class)
        """
        pass

    def save_weights(self):
        """Save actor and critic weights"""
        os.makedirs('model', exist_ok=True)
        tl.files.save_npz_dict(self.actor.trainable_weights, name='model/actor_weights.npz')
        tl.files.save_npz_dict(self.critic.trainable_weights, name='model/critic_weights.npz')

    def load_weights(self):
        """Load actor and critic weights"""
        tl.files.load_and_assign_npz_dict('model/actor_weights.npz', self.actor)
        tl.files.load_and_assign_npz_dict('model/critic_weights.npz', self.critic)

# ----------------- main loop -----------------
if __name__ == '__main__':
    # initialize environment
    env_id = 'Pendulum-v1'
    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high
    else:
        action_dim = env.action_space.n
        action_bound = None

    # initialize algorithm
    agent = Algorithm(state_dim, action_dim, action_bound)

    # ----------------- training loop -----------------
    if args.train:
        all_episode_rewards = []
        t0 = time.time()
        for episode in range(1, MAX_EPISODES + 1):
            state, _ = env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward

                # store transition & update (depends on algorithm)
                agent.update()

                state = next_state
                if done:
                    break

            all_episode_rewards.append(episode_reward)
            print(f'Episode: {episode}/{MAX_EPISODES} | Reward: {episode_reward:.4f} | Time: {time.time() - t0:.2f}s')

        # plot and save model
        plot_rewards(all_episode_rewards, algo_name='Algorithm', env_name=env_id)
        agent.save_weights()

    # ----------------- testing loop -----------------
    if args.test:
        agent.load_weights()
        t0 = time.time()
        for episode in range(1, 11):  # test 10 episodes
            state, _ = env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                action = agent.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
                if done:
                    break
            print(f'Test Episode: {episode}/10 | Reward: {episode_reward:.4f} | Time: {time.time() - t0:.2f}s')
