"""
Proximal Policy Optimization (PPO)
----------------------------
A simple version of Proximal Policy Optimization (PPO) using single thread.
Environment: OpenAI Gym Pendulum-v1
"""

import argparse
import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

#####################  hyper parameters  ####################
ENV_ID = 'Pendulum-v1'
RANDOM_SEED = 1
RENDER = False

ALG_NAME = 'PPO'
TRAIN_EPISODES = 10
TEST_EPISODES = 10
MAX_STEPS = 10
GAMMA = 0.9
LR_A = 0.0001
LR_C = 0.0002
BATCH_SIZE = 32
ACTOR_UPDATE_STEPS = 10
CRITIC_UPDATE_STEPS = 10

# ppo-penalty parameters
KL_TARGET = 0.01
LAM = 0.5

# ppo-clip parameters
EPSILON = 0.2

###############################  PPO  ####################################

class PPO(object):
    def __init__(self, state_dim, action_dim, action_bound, method='clip'):
        # critic
        with tf.name_scope('critic'):
            inputs = tl.layers.Input([None, state_dim], tf.float32, 'state')
            layer = tl.layers.Dense(64, tf.nn.relu)(inputs)
            layer = tl.layers.Dense(64, tf.nn.relu)(layer)
            v = tl.layers.Dense(1)(layer)
        self.critic = tl.models.Model(inputs, v)
        self.critic.train()

        # actor
        with tf.name_scope('actor'):
            inputs = tl.layers.Input([None, state_dim], tf.float32, 'state')
            layer = tl.layers.Dense(64, tf.nn.relu)(inputs)
            layer = tl.layers.Dense(64, tf.nn.relu)(layer)
            a = tl.layers.Dense(action_dim, tf.nn.tanh)(layer)
            mean = tl.layers.Lambda(lambda x: x * action_bound, name='lambda')(a)
            logstd = tf.Variable(np.zeros(action_dim, dtype=np.float32))
        self.actor = tl.models.Model(inputs, mean)
        self.actor.trainable_weights.append(logstd)
        self.actor.logstd = logstd
        self.actor.train()

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

        self.method = method
        if method == 'penalty':
            self.kl_target = KL_TARGET
            self.lam = LAM
        elif method == 'clip':
            self.epsilon = EPSILON

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer, self.cumulative_reward_buffer = [], []
        self.action_bound = action_bound

    def train_actor(self, state, action, adv, old_pi):
        with tf.GradientTape() as tape:
            mean, std = self.actor(state), tf.exp(self.actor.logstd)
            pi = tfp.distributions.Normal(mean, std)
            ratio = tf.exp(pi.log_prob(action) - old_pi.log_prob(action))
            surr = ratio * adv
            if self.method == 'penalty':
                kl = tfp.distributions.kl_divergence(old_pi, pi)
                kl_mean = tf.reduce_mean(kl)
                loss = -(tf.reduce_mean(surr - self.lam * kl))
            else:
                loss = -tf.reduce_mean(
                    tf.minimum(surr,
                               tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv)
                )
        grads = tape.gradient(loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_weights))
        if self.method == 'kl_pen':
            return kl_mean

    def train_critic(self, reward, state):
        reward = np.array(reward, dtype=np.float32)
        with tf.GradientTape() as tape:
            advantage = reward - self.critic(state)
            loss = tf.reduce_mean(tf.square(advantage))
        grads = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_weights))

    def update(self):
        s = np.array(self.state_buffer, np.float32)
        a = np.array(self.action_buffer, np.float32)
        r = np.array(self.cumulative_reward_buffer, np.float32)
        mean, std = self.actor(s), tf.exp(self.actor.logstd)
        pi = tfp.distributions.Normal(mean, std)
        adv = r - self.critic(s)

        if self.method == 'kl_pen':
            for _ in range(ACTOR_UPDATE_STEPS):
                kl = self.train_actor(s, a, adv, pi)
            if kl < self.kl_target / 1.5:
                self.lam /= 2
            elif kl > self.kl_target * 1.5:
                self.lam *= 2
        else:
            for _ in range(ACTOR_UPDATE_STEPS):
                self.train_actor(s, a, adv, pi)

        for _ in range(CRITIC_UPDATE_STEPS):
            self.train_critic(r, s)

        self.state_buffer.clear()
        self.action_buffer.clear()
        self.cumulative_reward_buffer.clear()
        self.reward_buffer.clear()

    def get_action(self, state, greedy=False):
        state = state[np.newaxis, :].astype(np.float32)
        mean, std = self.actor(state), tf.exp(self.actor.logstd)
        if greedy:
            action = mean[0]
        else:
            pi = tfp.distributions.Normal(mean, std)
            action = tf.squeeze(pi.sample(1), axis=0)[0]
        return np.clip(action, -self.action_bound, self.action_bound)

    def save(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        os.makedirs(path, exist_ok=True)
        tl.files.save_npz_dict(self.actor.trainable_weights, name=os.path.join(path, 'actor_weights.npz'))
        tl.files.save_npz_dict(self.critic.trainable_weights, name=os.path.join(path, 'critic_weights.npz'))

    def load(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_and_assign_npz_dict(name=os.path.join(path, 'actor_weights.npz'), network=self.actor)
        tl.files.load_and_assign_npz_dict(name=os.path.join(path, 'critic_weights.npz'), network=self.critic)



    def store_transition(self, state, action, reward):
        self.state_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def finish_path(self, next_state, done):
        if done:
            v_s_ = 0
        else:
            v_s_ = self.critic(np.array([next_state], np.float32))[0, 0]
        discounted_r = []
        for r in self.reward_buffer[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        discounted_r = np.array(discounted_r)[:, np.newaxis]
        self.cumulative_reward_buffer.extend(discounted_r)
        self.reward_buffer.clear()


if __name__ == '__main__':
    env = gym.make(ENV_ID)

    # seed environment and spaces
    state, info = env.reset(seed=RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)
    env.observation_space.seed(RANDOM_SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    agent = PPO(state_dim, action_dim, action_bound)

    t0 = time.time()
    if args.train:
        all_episode_reward = []
        for episode in range(TRAIN_EPISODES):
            state, info = env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                if RENDER:
                    env.render()
                action = agent.get_action(state)
                state_, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.store_transition(state, action, reward)
                state = state_
                episode_reward += reward

                if len(agent.state_buffer) >= BATCH_SIZE:
                    agent.finish_path(state_, done)
                    agent.update()
                if done:
                    break
            agent.finish_path(state_, done)
            print(
                'Training  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TRAIN_EPISODES, episode_reward, time.time() - t0)
            )
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
        agent.save()

        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

    if args.test:
        agent.load()
        for episode in range(TEST_EPISODES):
            state, info = env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                if RENDER:
                    env.render()
                state_, reward, terminated, truncated, info = env.step(agent.get_action(state, greedy=True))
                done = terminated or truncated
                state = state_
                episode_reward += reward
                if done:
                    break
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TEST_EPISODES, episode_reward, time.time() - t0))
