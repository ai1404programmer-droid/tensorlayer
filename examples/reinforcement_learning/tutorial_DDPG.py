"""
Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------
An algorithm concurrently learns a Q-function and a policy.
It uses off-policy data and the Bellman equation to learn the Q-function,
and uses the Q-function to learn the policy.

Reference
---------
Deterministic Policy Gradient Algorithms, Silver et al. 2014
Continuous Control With Deep Reinforcement Learning, Lillicrap et al. 2016
MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/

Environment
-----------
Openai Gym Pendulum-v0, continual action space

Prerequisites
-------------
tensorflow >=2.0.0a0
tensorflow-proactionsbility 0.6.0
tensorlayer >=2.0.0

To run
------
python tutorial_DDPG.py --train/test

"""
"""
Deep Deterministic Policy Gradient (DDPG) - Gymnasium compatible
---------------------------------------------------------------
Continuous action space RL algorithm
"""

import argparse
import os
import time
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl

# ------------------- Arguments -------------------
parser = argparse.ArgumentParser(description='Train or test DDPG agent.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

# ------------------- Hyperparameters -------------------
ENV_ID = 'Pendulum-v1'  # Gymnasium uses Pendulum-v1
RANDOM_SEED = 2
RENDER = False

ALG_NAME = 'DDPG'
TRAIN_EPISODES = 10
TEST_EPISODES = 10
MAX_STEPS = 20

LR_A = 0.001
LR_C = 0.002
GAMMA = 0.9
TAU = 0.01
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32
VAR = 2  # exploration noise

# ------------------- DDPG Agent -------------------
class DDPG(object):
    def __init__(self, action_dim, state_dim, action_range):
        self.memory = np.zeros((MEMORY_CAPACITY, state_dim*2 + action_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.action_dim, self.state_dim, self.action_range = action_dim, state_dim, action_range
        self.var = VAR

        W_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        b_init = tf.constant_initializer(0.1)

        # Actor Network
        def get_actor(input_shape, name=''):
            inputs = tl.layers.Input(input_shape, name='A_input')
            x = tl.layers.Dense(64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l1')(inputs)
            x = tl.layers.Dense(64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A_l2')(x)
            x = tl.layers.Dense(self.action_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_a')(x)
            x = tl.layers.Lambda(lambda x: self.action_range * x)(x)
            return tl.models.Model(inputs=inputs, outputs=x, name='Actor' + name)

        # Critic Network
        def get_critic(input_state_shape, input_action_shape, name=''):
            s_input = tl.layers.Input(input_state_shape, name='C_s_input')
            a_input = tl.layers.Input(input_action_shape, name='C_a_input')
            x = tl.layers.Concat(1)([s_input, a_input])
            x = tl.layers.Dense(64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = tl.layers.Dense(64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l2')(x)
            x = tl.layers.Dense(1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return tl.models.Model(inputs=[s_input, a_input], outputs=x, name='Critic' + name)

        # Initialize networks
        self.actor = get_actor([None, state_dim])
        self.critic = get_critic([None, state_dim], [None, action_dim])
        self.actor.train()
        self.critic.train()

        # Target networks
        def copy_para(from_model, to_model):
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        self.actor_target = get_actor([None, state_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        self.critic_target = get_critic([None, state_dim], [None, action_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.ema = tf.train.ExponentialMovingAverage(decay=1-TAU)

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

    def ema_update(self):
        paras = self.actor.trainable_weights + self.critic.trainable_weights
        self.ema.apply(paras)
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))

    def get_action(self, s, greedy=False):
        a = self.actor(np.array([s], dtype=np.float32))[0]
        if greedy:
            return a
        return np.clip(np.random.normal(a, self.var), -self.action_range, self.action_range)

    def learn(self):
        self.var *= .9995
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        datas = self.memory[indices, :]
        states = datas[:, :self.state_dim]
        actions = datas[:, self.state_dim:self.state_dim+self.action_dim]
        rewards = datas[:, -self.state_dim-1:-self.state_dim]
        states_ = datas[:, -self.state_dim:]

        # Critic update
        with tf.GradientTape() as tape:
            actions_ = self.actor_target(states_)
            q_ = self.critic_target([states_, actions_])
            y = rewards + GAMMA * q_
            q = self.critic([states, actions])
            td_error = tf.losses.mean_squared_error(y, q)
        critic_grads = tape.gradient(td_error, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_weights))

        # Actor update
        with tf.GradientTape() as tape:
            a = self.actor(states)
            q = self.critic([states, a])
            actor_loss = -tf.reduce_mean(q)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor.trainable_weights))
        self.ema_update()

    def store_transition(self, s, a, r, s_):
        s = s.astype(np.float32)
        s_ = s_.astype(np.float32)
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1


    def save(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        os.makedirs(path, exist_ok=True)

        tl.files.save_npz_dict(self.actor.trainable_weights, name=os.path.join(path, 'model_actor.npz'))
        tl.files.save_npz_dict(self.actor_target.trainable_weights, name=os.path.join(path, 'model_actor_target.npz'))
        tl.files.save_npz_dict(self.critic.trainable_weights, name=os.path.join(path, 'model_critic.npz'))
        tl.files.save_npz_dict(self.critic_target.trainable_weights, name=os.path.join(path, 'model_critic_target.npz'))
    def load(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))

        tl.files.load_and_assign_npz_dict(name=os.path.join(path, 'model_actor.npz'), network=self.actor)
        tl.files.load_and_assign_npz_dict(name=os.path.join(path, 'model_actor_target.npz'), network=self.actor_target)
        tl.files.load_and_assign_npz_dict(name=os.path.join(path, 'model_critic.npz'), network=self.critic)
        tl.files.load_and_assign_npz_dict(name=os.path.join(path, 'model_critic_target.npz'), network=self.critic_target)

# ------------------- Main -------------------
if __name__ == '__main__':
    env = gym.make(ENV_ID)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high

    agent = DDPG(action_dim, state_dim, action_range)

    t0 = time.time()

    # Train
    if args.train:
        all_episode_reward = []
        for episode in range(TRAIN_EPISODES):
            state, _ = env.reset(seed=RANDOM_SEED)
            episode_reward = 0
            for step in range(MAX_STEPS):
                if RENDER:
                    env.render()
                action = agent.get_action(state)
                state_, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.store_transition(state, action, reward, state_)
                if agent.pointer > MEMORY_CAPACITY:
                    agent.learn()
                state = state_
                episode_reward += reward
                if done:
                    break
            all_episode_reward.append(episode_reward if episode==0 else all_episode_reward[-1]*0.9 + episode_reward*0.1)
            print(f'Training | Episode: {episode+1}/{TRAIN_EPISODES} | Reward: {episode_reward:.4f} | Time: {time.time()-t0:.2f}')

        agent.save()
        plt.plot(all_episode_reward)
        os.makedirs('image', exist_ok=True)
        plt.savefig(os.path.join('image', f'{ALG_NAME}_{ENV_ID}.png'))

    # Test
    if args.test:
        agent.load()
        for episode in range(TEST_EPISODES):
            state, _ = env.reset(seed=RANDOM_SEED)
            episode_reward = 0
            for step in range(MAX_STEPS):
                env.render()
                action = agent.get_action(state, greedy=True)
                state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                if done:
                    break
            print(f'Testing | Episode: {episode+1}/{TEST_EPISODES} | Reward: {episode_reward:.4f} | Time: {time.time()-t0:.2f}')
