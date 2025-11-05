"""
Soft Actor-Critic (SAC) - TensorLayer
-------------------------------------
- Train / Test
- Save / Load weights using tl.files.save_npz / tl.files.load_and_assign_npz
- Compatible with gymnasium Pendulum-v1
Requirements:
  tensorflow, tensorlayer, tensorflow-probability, gymnasium
"""

import argparse
import os
import random
import time
import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl
from tensorlayer.layers import Dense
from tensorlayer.models import Model
import matplotlib.pyplot as plt

Normal = tfp.distributions.Normal
tl.logging.set_verbosity(tl.logging.DEBUG)

# ------------------ Arguments ------------------
parser = argparse.ArgumentParser(description='Train or test SAC agent.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
parser.add_argument('--env', dest='env', default='Pendulum-v1')
parser.add_argument('--save_dir', dest='save_dir', default='model/SAC_Pendulum-v1')
args = parser.parse_args()

# ------------------ Hyperparameters ------------------
ENV_ID = args.env
RANDOM_SEED = 2
RENDER = False

ALG_NAME = 'SAC'
TRAIN_EPISODES = 10
TEST_EPISODES = 10
MAX_STEPS = 10
EXPLORE_STEPS = 10

BATCH_SIZE = 256
HIDDEN_DIM = 64
UPDATE_ITR = 3
SOFT_Q_LR = 3e-4
POLICY_LR = 3e-4
ALPHA_LR = 3e-4
REWARD_SCALE = 1.0
REPLAY_BUFFER_SIZE = int(5e5)
AUTO_ENTROPY = True
TARGET_ENTROPY = None  # if None, -action_dim
SAVE_DIR = args.save_dir

# ------------------ Replay Buffer ------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state.astype(np.float32), action.astype(np.float32), reward.astype(np.float32), next_state.astype(np.float32), done.astype(np.float32)

    def __len__(self):
        return len(self.buffer)

# ------------------ Networks ------------------
class SoftQNetwork(Model):
    def __init__(self, num_inputs, num_actions, hidden_dim, name='soft_q'):
        super(SoftQNetwork, self).__init__(name=name)
        input_dim = num_inputs + num_actions
        w_init = tf.keras.initializers.GlorotNormal()
        self.l1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=input_dim, name=name+'_l1')
        self.l2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name=name+'_l2')
        self.out = Dense(n_units=1, W_init=w_init, in_channels=hidden_dim, name=name+'_out')

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return self.out(x)

class PolicyNetwork(Model):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_range=1.0, name='policy'):
        super(PolicyNetwork, self).__init__(name=name)
        self.action_range = float(action_range)
        self.num_actions = int(num_actions)
        w_init = tf.keras.initializers.GlorotNormal()
        self.l1 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=num_inputs, name=name+'_l1')
        self.l2 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name=name+'_l2')
        self.l3 = Dense(n_units=hidden_dim, act=tf.nn.relu, W_init=w_init, in_channels=hidden_dim, name=name+'_l3')
        self.mean_linear = Dense(n_units=num_actions, W_init=w_init, in_channels=hidden_dim, name=name+'_mean')
        self.log_std_linear = Dense(n_units=num_actions, W_init=w_init, in_channels=hidden_dim, name=name+'_logstd')
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        x = self.l1(state)
        x = self.l2(x)
        x = self.l3(x)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = tf.clip_by_value(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = tf.exp(log_std)
        normal = Normal(0., 1.)
        z = normal.sample(mean.shape)
        pre_tanh = mean + std * z
        action_0 = tf.tanh(pre_tanh)
        action = self.action_range * action_0
        log_prob = Normal(mean, std).log_prob(pre_tanh) - tf.math.log(1.0 - action_0**2 + epsilon)
        log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
        return action, log_prob, z, mean, log_std

    def get_action(self, state, greedy=False):
        if isinstance(state, np.ndarray):
            s = tf.convert_to_tensor(state.astype(np.float32)[None, :])
        else:
            s = tf.convert_to_tensor(state[None, :], dtype=tf.float32)
        mean, log_std = self.forward(s)
        if greedy:
            return (tf.tanh(mean) * self.action_range).numpy()[0]
        std = tf.exp(log_std)
        normal = Normal(0., 1.)
        z = normal.sample(mean.shape)
        pre_tanh = mean + std * z
        return (tf.tanh(pre_tanh) * self.action_range).numpy()[0]

    def sample_action(self):
        a = tf.random.uniform([self.num_actions], -1, 1)
        return (self.action_range * a).numpy()

# ------------------ SAC Agent ------------------
class SACAgent:
    def __init__(self, state_dim, action_dim, action_range, hidden_dim, replay_buffer):
        self.replay_buffer = replay_buffer
        self.soft_q1 = SoftQNetwork(state_dim, action_dim, hidden_dim, name='q1')
        self.soft_q2 = SoftQNetwork(state_dim, action_dim, hidden_dim, name='q2')
        self.target_q1 = SoftQNetwork(state_dim, action_dim, hidden_dim, name='target_q1')
        self.target_q2 = SoftQNetwork(state_dim, action_dim, hidden_dim, name='target_q2')
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim, action_range, name='policy')

        # Dummy forward to initialize weights (do not pass is_train)
        dummy_state = tf.zeros((1, state_dim), dtype=tf.float32)
        dummy_action = tf.zeros((1, action_dim), dtype=tf.float32)
        dummy_input = tf.concat([dummy_state, dummy_action], axis=1)

        self.soft_q1.train(); _ = self.soft_q1(dummy_input)
        self.soft_q2.train(); _ = self.soft_q2(dummy_input)
        self.target_q1.eval(); _ = self.target_q1(dummy_input)
        self.target_q2.eval(); _ = self.target_q2(dummy_input)
        self.policy.train(); _ = self.policy(dummy_state)

        self._hard_update(self.soft_q1, self.target_q1)
        self._hard_update(self.soft_q2, self.target_q2)

        self.soft_q_opt1 = tf.optimizers.Adam(SOFT_Q_LR)
        self.soft_q_opt2 = tf.optimizers.Adam(SOFT_Q_LR)
        self.policy_opt = tf.optimizers.Adam(POLICY_LR)
        self.log_alpha = tf.Variable(0., dtype=tf.float32)
        self.alpha_opt = tf.optimizers.Adam(ALPHA_LR)
        self.alpha = tf.exp(self.log_alpha)

    def _hard_update(self, src_net, tgt_net):
        for s, t in zip(src_net.trainable_weights, tgt_net.trainable_weights):
            t.assign(s)

    def _soft_update(self, src_net, tgt_net, tau):
        for s, t in zip(src_net.trainable_weights, tgt_net.trainable_weights):
            t.assign(t * (1 - tau) + s * tau)

    def update(self, batch_size=BATCH_SIZE, reward_scale=REWARD_SCALE, auto_entropy=AUTO_ENTROPY,
               target_entropy=None, gamma=0.99, tau=1e-2):

        if target_entropy is None:
            target_entropy = -1.0 * self.policy.num_actions

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        reward = reward[:, None]; done = done[:, None]
        reward = reward_scale * (reward - np.mean(reward)) / (np.std(reward) + 1e-6)

        new_next_action, next_log_prob, _, _, _ = self.policy.evaluate(next_state)
        target_input = tf.concat([next_state, new_next_action], axis=1)
        target_q1 = self.target_q1(target_input)
        target_q2 = self.target_q2(target_input)
        target_q_min = tf.minimum(target_q1, target_q2) - self.alpha * next_log_prob
        target_q = reward + (1 - done) * gamma * target_q_min

        q_input = tf.concat([state, action], axis=1)

        with tf.GradientTape() as tape:
            q1_pred = self.soft_q1(q_input)
            loss_q1 = tf.reduce_mean(tf.square(q1_pred - target_q))
        grads = tape.gradient(loss_q1, self.soft_q1.trainable_weights)
        self.soft_q_opt1.apply_gradients(zip(grads, self.soft_q1.trainable_weights))

        with tf.GradientTape() as tape:
            q2_pred = self.soft_q2(q_input)
            loss_q2 = tf.reduce_mean(tf.square(q2_pred - target_q))
        grads = tape.gradient(loss_q2, self.soft_q2.trainable_weights)
        self.soft_q_opt2.apply_gradients(zip(grads, self.soft_q2.trainable_weights))

        with tf.GradientTape() as tape:
            new_action, log_prob, _, _, _ = self.policy.evaluate(state)
            new_q_input = tf.concat([state, new_action], axis=1)
            q1_new = self.soft_q1(new_q_input)
            q2_new = self.soft_q2(new_q_input)
            q_new = tf.minimum(q1_new, q2_new)
            policy_loss = tf.reduce_mean(self.alpha * log_prob - q_new)
        grads = tape.gradient(policy_loss, self.policy.trainable_weights)
        self.policy_opt.apply_gradients(zip(grads, self.policy.trainable_weights))

        if auto_entropy:
            with tf.GradientTape() as tape:
                alpha_loss = -tf.reduce_mean(self.log_alpha * (log_prob + target_entropy))
            grad = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_opt.apply_gradients(zip(grad, [self.log_alpha]))
            self.alpha = tf.exp(self.log_alpha)

        self._soft_update(self.soft_q1, self.target_q1, tau)
        self._soft_update(self.soft_q2, self.target_q2, tau)

    # Save / Load
    def save(self, path=SAVE_DIR):
        os.makedirs(path, exist_ok=True)
        extend = lambda s: os.path.join(path, s)

        def save_model_weights(model, fname):
            weights = {f'param_{i}': w.numpy() for i, w in enumerate(model.trainable_weights)}
            np.savez(extend(fname), **weights)

        save_model_weights(self.soft_q1, 'q1.npz')
        save_model_weights(self.soft_q2, 'q2.npz')
        save_model_weights(self.target_q1, 'tq1.npz')
        save_model_weights(self.target_q2, 'tq2.npz')
        save_model_weights(self.policy, 'policy.npz')

        np.save(extend('log_alpha.npy'), self.log_alpha.numpy())

    def load(self, path=SAVE_DIR):
        extend = lambda s: os.path.join(path, s)

        def load_model_weights(model, fname):
            data = np.load(extend(fname))
            for i, w in enumerate(model.trainable_weights):
                w.assign(data[f'param_{i}'])

        load_model_weights(self.soft_q1, 'q1.npz')
        load_model_weights(self.soft_q2, 'q2.npz')
        load_model_weights(self.target_q1, 'tq1.npz')
        load_model_weights(self.target_q2, 'tq2.npz')
        load_model_weights(self.policy, 'policy.npz')

        self.log_alpha.assign(np.load(extend('log_alpha.npy')))
        self.alpha = tf.exp(self.log_alpha)

# ------------------ Main ------------------
def main():
    env = gym.make(ENV_ID)
    random.seed(RANDOM_SEED); np.random.seed(RANDOM_SEED); tf.random.set_seed(RANDOM_SEED)
    try:
        env.action_space.seed(RANDOM_SEED)
        env.observation_space.seed(RANDOM_SEED)
    except:
        pass

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_range = float(env.action_space.high[0])

    replay = ReplayBuffer(REPLAY_BUFFER_SIZE)
    agent = SACAgent(state_dim, action_dim, action_range, HIDDEN_DIM, replay)

    t0 = time.time()
    rewards_curve = []

    if args.train:
        frame_idx = 0
        for ep in range(TRAIN_EPISODES):
            state, _ = env.reset(seed=RANDOM_SEED + ep)
            state = state.astype(np.float32)
            ep_reward = 0.0
            for step in range(MAX_STEPS):
                if RENDER: env.render()
                if frame_idx > EXPLORE_STEPS:
                    action = agent.policy.get_action(state)
                else:
                    action = agent.policy.sample_action()
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_state = next_state.astype(np.float32)
                replay.push(state, action, float(reward), next_state, float(done))
                state = next_state
                ep_reward += reward
                frame_idx += 1

                if len(replay) > BATCH_SIZE:
                    for _ in range(UPDATE_ITR):
                        agent.update(BATCH_SIZE, REWARD_SCALE, AUTO_ENTROPY, TARGET_ENTROPY)

                if done: break

            rewards_curve.append(ep_reward if ep == 0 else rewards_curve[-1]*0.9 + ep_reward*0.1)
            print(f'Train | Ep {ep+1}/{TRAIN_EPISODES} | Reward {ep_reward:.3f} | Time {time.time()-t0:.2f}s')

        agent.save(SAVE_DIR)
        os.makedirs('image', exist_ok=True)
        plt.plot(rewards_curve)
        plt.savefig(os.path.join('image', f'{ALG_NAME}_{ENV_ID}.png'))
        print('Training finished and model saved to', SAVE_DIR)

    if args.test:
        try:
            agent.load(SAVE_DIR)
            print('Loaded weights from', SAVE_DIR)
        except Exception as e:
            print('Warning: could not load weights:', e)

        for ep in range(TEST_EPISODES):
            state, _ = env.reset(seed=RANDOM_SEED + ep + 1000)
            state = state.astype(np.float32)
            ep_reward = 0.0
            for step in range(MAX_STEPS):
                env.render()
                action = agent.policy.get_action(state, greedy=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state.astype(np.float32)
                ep_reward += reward
                if done: break
            print(f'Test  | Ep {ep+1}/{TEST_EPISODES} | Reward {ep_reward:.3f} | Time {time.time()-t0:.2f}s')

if __name__ == '__main__':
    main()
