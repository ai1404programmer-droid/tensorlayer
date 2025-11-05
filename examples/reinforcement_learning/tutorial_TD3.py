"""
Twin Delayed DDPG (TD3) - نسخه کامل و اصلاح‌شده با TensorLayer
----------------------------------------------------------------
حل خطاهای:
- Unknown variable در optimizer (Critic)
- train/eval mode not set
- سایر خطاهای TensorLayer/TensorFlow 2.x
"""
import argparse
import os
import random
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl
from tensorlayer.layers import Dense, Input, Lambda

Normal = tfp.distributions.Normal
tl.logging.set_verbosity(tl.logging.DEBUG)

parser = argparse.ArgumentParser(description='Train or test TD3 on Pendulum-v1')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

##################### Hyperparameters ####################
ENV_ID = 'Pendulum-v1'
RANDOM_SEED = 2
RENDER = False

ALG_NAME = 'TD3'
TRAIN_EPISODES = 10
TEST_EPISODES = 5
MAX_STEPS = 10
BATCH_SIZE = 64
EXPLORE_STEPS = 10

HIDDEN_DIM = 64
UPDATE_ITR = 3
Q_LR = 3e-4
POLICY_LR = 3e-4
EXPLORE_NOISE_SCALE = 0.1
EVAL_NOISE_SCALE = 0.2
REWARD_SCALE = 1.0
REPLAY_BUFFER_SIZE = int(5e5)

######################## Replay Buffer ########################
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state.astype(np.float32), action.astype(np.float32), reward.astype(np.float32), \
               next_state.astype(np.float32), done.astype(np.float32)

    def __len__(self):
        return len(self.buffer)

######################## Neural Networks ########################
class QNetwork:
    def __init__(self, num_inputs, num_actions, hidden_dim, init_w=3e-3, name='QNetwork'):
        input_dim = num_inputs + num_actions
        w_init = tf.random_uniform_initializer(-init_w, init_w)
        ni = Input([None, input_dim], name=f'{name}_input')
        nn = Dense(hidden_dim, act=tf.nn.relu, W_init=w_init)(ni)
        nn = Dense(hidden_dim, act=tf.nn.relu, W_init=w_init)(nn)
        nn = Dense(1, W_init=w_init)(nn)
        self.model = tl.models.Model(inputs=ni, outputs=nn, name=f'{name}_model')

    def __call__(self, x):
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        return self.model(x)

class PolicyNetwork:
    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1., init_w=3e-3, name='PolicyNetwork'):
        w_init = tf.random_uniform_initializer(-init_w, init_w)
        ni = Input([None, num_inputs], name=f'{name}_input')
        nn = Dense(hidden_dim, act=tf.nn.relu, W_init=w_init)(ni)
        nn = Dense(hidden_dim, act=tf.nn.relu, W_init=w_init)(nn)
        nn = Dense(hidden_dim, act=tf.nn.relu, W_init=w_init)(nn)
        nn = Dense(num_actions, W_init=w_init,
                   b_init=tf.random_uniform_initializer(-init_w, init_w))(nn)
        out = Lambda(lambda x: tf.nn.tanh(x), name=f'{name}_tanh')(nn)
        self.model = tl.models.Model(inputs=ni, outputs=out, name=f'{name}_model')
        self.action_range = action_range
        self.num_actions = num_actions

    def forward(self, state):
        if not isinstance(state, tf.Tensor):
            state = tf.convert_to_tensor(state, dtype=tf.float32)
        return self.model(state)

    def get_action(self, state, explore_noise_scale=0.0, greedy=False):
        self.model.eval()
        if not isinstance(state, tf.Tensor):
            state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = self.model(state)
        action = (self.action_range * action).numpy()[0]
        if greedy:
            return action
        noise = np.random.normal(0, explore_noise_scale, size=action.shape)
        return np.clip(action + noise, -self.action_range, self.action_range)

    def sample_action(self):
        a = tf.random.uniform([self.num_actions], -1, 1)
        return self.action_range * a.numpy()

######################## TD3 Algorithm ########################
class TD3:
    def __init__(self, state_dim, action_dim, action_range, hidden_dim, replay_buffer,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                 actor_lr=3e-4, critic_lr=3e-4, tau=0.005, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.replay_buffer = replay_buffer

        # Networks
        self.q_net1 = QNetwork(state_dim, action_dim, hidden_dim, name='QNet1')
        self.q_net2 = QNetwork(state_dim, action_dim, hidden_dim, name='QNet2')
        self.target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim, name='TargetQNet1')
        self.target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim, name='TargetQNet2')

        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range, name='PolicyNet')
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range, name='TargetPolicyNet')

        # Initialize targets
        self.target_ini(self.q_net1.model, self.target_q_net1.model)
        self.target_ini(self.q_net2.model, self.target_q_net2.model)
        self.target_ini(self.policy_net.model, self.target_policy_net.model)

        # Optimizers (دو optimizer جدا برای Q1 و Q2)
        self.actor_optimizer = tf.optimizers.Adam(actor_lr)
        self.q1_optimizer = tf.optimizers.Adam(critic_lr)
        self.q2_optimizer = tf.optimizers.Adam(critic_lr)

        # Hyperparams
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.tau = tau
        self.gamma = gamma
        self.total_it = 0

    def target_ini(self, src_model, tgt_model):
        for tgt_w, src_w in zip(tgt_model.trainable_weights, src_model.trainable_weights):
            tgt_w.assign(src_w)

    def target_soft_update(self, src_model, tgt_model, soft_tau):
        for tgt_w, src_w in zip(tgt_model.trainable_weights, src_model.trainable_weights):
            tgt_w.assign(tgt_w * (1 - soft_tau) + src_w * soft_tau)

    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        # set modes
        self.q_net1.model.train()
        self.q_net2.model.train()
        self.policy_net.model.train()
        self.target_q_net1.model.eval()
        self.target_q_net2.model.eval()
        self.target_policy_net.model.eval()

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)[:, None]
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)[:, None]

        noise = tf.clip_by_value(
            tf.random.normal([batch_size, self.action_dim], stddev=self.policy_noise),
            -self.noise_clip, self.noise_clip
        )
        next_actions = self.target_policy_net.forward(next_states)
        next_actions = tf.clip_by_value(next_actions + noise, -self.action_range, self.action_range)

        # Target Q
        target_q1 = self.target_q_net1.model(tf.concat([next_states, next_actions], axis=1))
        target_q2 = self.target_q_net2.model(tf.concat([next_states, next_actions], axis=1))
        target_q = tf.minimum(target_q1, target_q2)
        target_q = rewards + (1.0 - dones) * self.gamma * target_q

        # Current Q & Critic Update
        with tf.GradientTape(persistent=True) as tape:
            current_q1 = self.q_net1.model(tf.concat([states, actions], axis=1))
            current_q2 = self.q_net2.model(tf.concat([states, actions], axis=1))
            critic_loss = tf.reduce_mean(tf.square(current_q1 - target_q) + tf.square(current_q2 - target_q))

        grads1 = tape.gradient(critic_loss, self.q_net1.model.trainable_weights)
        grads2 = tape.gradient(critic_loss, self.q_net2.model.trainable_weights)
        self.q1_optimizer.apply_gradients(zip(grads1, self.q_net1.model.trainable_weights))
        self.q2_optimizer.apply_gradients(zip(grads2, self.q_net2.model.trainable_weights))
        del tape

        # Delayed Policy Update
        if self.total_it % self.policy_delay == 0:
            with tf.GradientTape() as tape:
                new_actions = self.policy_net.model(states)
                actor_loss = -tf.reduce_mean(self.q_net1.model(tf.concat([states, new_actions], axis=1)))
            actor_grads = tape.gradient(actor_loss, self.policy_net.model.trainable_weights)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.policy_net.model.trainable_weights))

            # soft update targets
            self.target_soft_update(self.q_net1.model, self.target_q_net1.model, self.tau)
            self.target_soft_update(self.q_net2.model, self.target_q_net2.model, self.tau)
            self.target_soft_update(self.policy_net.model, self.target_policy_net.model, self.tau)

        self.total_it += 1

########################## Main ##########################
if __name__ == '__main__':
    env = gym.make(ENV_ID)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = env.action_space.high[0]

    env.reset(seed=RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    agent = TD3(state_dim, action_dim, action_range, HIDDEN_DIM, replay_buffer,
                policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                actor_lr=POLICY_LR, critic_lr=Q_LR, tau=1e-2, gamma=0.99)
    t0 = time.time()

    if args.train:
        all_episode_reward = []
        frame_idx = 0
        for episode in range(TRAIN_EPISODES):
            state, _ = env.reset()
            state = state.astype(np.float32)
            episode_reward = 0
            for step in range(MAX_STEPS):
                if RENDER: env.render()
                if frame_idx > EXPLORE_STEPS:
                    action = agent.policy_net.get_action(state, EXPLORE_NOISE_SCALE)
                else:
                    action = agent.policy_net.sample_action()
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = next_state.astype(np.float32)
                replay_buffer.push(state, action, reward, next_state, float(done))
                state = next_state
                episode_reward += reward
                frame_idx += 1
                if len(replay_buffer) > BATCH_SIZE:
                    for _ in range(UPDATE_ITR):
                        agent.update(BATCH_SIZE)
                if done: break
            all_episode_reward.append(episode_reward)
            print(f'Episode {episode+1}/{TRAIN_EPISODES}, Reward: {episode_reward:.3f}, Time: {time.time()-t0:.2f}s')

        plt.plot(all_episode_reward)
        os.makedirs('image', exist_ok=True)
        plt.savefig(os.path.join('image', f'{ALG_NAME}_{ENV_ID}.png'))

    if args.test:
        for episode in range(TEST_EPISODES):
            state, _ = env.reset()
            state = state.astype(np.float32)
            episode_reward = 0
            for step in range(MAX_STEPS):
                if RENDER: env.render()
                action = agent.policy_net.get_action(state, explore_noise_scale=0.0, greedy=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = next_state.astype(np.float32)
                state = next_state
                episode_reward += reward
                if done: break
            print(f'[TEST] Episode {episode+1}/{TEST_EPISODES} | Reward: {episode_reward:.3f}')
