import argparse
import copy
import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test TRPO agent.')
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=True)
args = parser.parse_args()

##################### Hyperparameters ####################
ENV_ID = 'Pendulum-v1'
RANDOM_SEED = 2
RENDER = False

ALG_NAME = 'TRPO'
TRAIN_EPISODES = 10
TEST_EPISODES = 10
MAX_STEPS = 10

HIDDEN_SIZES = [64, 64]
GAMMA = 0.99
DELTA = 0.01
VF_LR = 1e-3
TRAIN_VF_ITERS = 100
DAMPING_COEFF = 0.1
CG_ITERS = 10
BACKTRACK_ITERS = 10
BACKTRACK_COEFF = 0.8
LAM = 0.97
SAVE_FREQ = 10
EPS = 1e-8
BATCH_SIZE = 512

##################### Buffer ####################
class GAE_Buffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.mean_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.log_std_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, mean, log_std):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.mean_buf[self.ptr] = mean
        self.log_std_buf[self.ptr] = log_std
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = self._discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def _discount_cumsum(self, x, discount):
        return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]

    def is_full(self):
        return self.ptr == self.max_size

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        return (self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf,
                self.logp_buf, self.mean_buf, self.log_std_buf)

##################### TRPO Agent ####################
class TRPO:
    def __init__(self, state_dim, action_dim, action_bound):
        # Critic
        layer = input_layer = tl.layers.Input([None, state_dim], tf.float32)
        for d in HIDDEN_SIZES:
            layer = tl.layers.Dense(d, act=tf.nn.relu)(layer)
        v = tl.layers.Dense(1)(layer)
        self.critic = tl.models.Model(input_layer, v)
        self.critic.train()

        # Actor
        layer = input_layer = tl.layers.Input([None, state_dim], tf.float32)
        for d in HIDDEN_SIZES:
            layer = tl.layers.Dense(d, act=tf.nn.relu)(layer)
        mean = tl.layers.Dense(action_dim, act=tf.nn.tanh)(layer)
        mean = tl.layers.Lambda(lambda x: x * action_bound)(mean)
        log_std = tf.Variable(np.zeros(action_dim, dtype=np.float32))

        self.actor = tl.models.Model(input_layer, mean)
        self.actor.trainable_weights.append(log_std)
        self.actor.log_std = log_std
        self.actor.train()

        self.buf = GAE_Buffer(state_dim, action_dim, BATCH_SIZE, GAMMA, LAM)
        self.critic_optimizer = tf.optimizers.Adam(VF_LR)
        self.action_bound = action_bound

    def get_action(self, state, greedy=False):
        state = np.array([state], np.float32)
        mean = self.actor(state)
        std = tf.exp(self.actor.log_std) * tf.ones_like(mean)
        pi = tfp.distributions.Normal(mean, std)
        action = mean if greedy else pi.sample()
        logp_pi = tf.reduce_sum(pi.log_prob(action), axis=1)
        action = np.clip(action, -self.action_bound, self.action_bound)
        value = self.critic(state)
        return action[0], value[0, 0], logp_pi[0], mean[0], self.actor.log_std

    def pi_loss(self, states, actions, adv, old_log_prob):
        mean = self.actor(states)
        pi = tfp.distributions.Normal(mean, tf.exp(self.actor.log_std))
        log_prob = tf.reduce_sum(pi.log_prob(actions), axis=1)
        ratio = tf.exp(log_prob - old_log_prob)
        surr = tf.reduce_mean(ratio * adv)
        return -surr

    def gradient(self, states, actions, adv, old_log_prob):
        pi_params = self.actor.trainable_weights
        with tf.GradientTape() as tape:
            loss = self.pi_loss(states, actions, adv, old_log_prob)
        grad = tape.gradient(loss, pi_params)
        return self._flat_concat(grad), loss

    def train_vf(self, states, rewards_to_go):
        for _ in range(TRAIN_VF_ITERS):
            with tf.GradientTape() as tape:
                value = self.critic(states)
                loss = tf.reduce_mean((rewards_to_go - tf.squeeze(value, axis=1))**2)
            grads = tape.gradient(loss, self.critic.trainable_weights)
            self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_weights))

    def kl(self, states, old_mean, old_log_std):
        old_mean = old_mean
        old_log_std = old_log_std
        old_std = tf.exp(old_log_std)
        old_pi = tfp.distributions.Normal(old_mean, old_std)

        mean = self.actor(states)
        std = tf.exp(self.actor.log_std) * tf.ones_like(mean)
        pi = tfp.distributions.Normal(mean, std)

        kl = tfp.distributions.kl_divergence(pi, old_pi)
        return tf.reduce_mean(tf.reduce_sum(kl, axis=1))

    def _flat_concat(self, xs):
        return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)

    def get_pi_params(self):
        return self._flat_concat(self.actor.trainable_weights)

    def set_pi_params(self, flat_params):
        pi_params = self.actor.trainable_weights
        sizes = [tf.size(p) for p in pi_params]
        splits = tf.split(flat_params, sizes)
        for p, s in zip(pi_params, splits):
            p.assign(tf.reshape(s, p.shape))

    def hvp(self, states, old_mean, old_log_std, x):
        pi_params = self.actor.trainable_weights
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape0:
                d_kl = self.kl(states, old_mean, old_log_std)
            g = self._flat_concat(tape0.gradient(d_kl, pi_params))
            l = tf.reduce_sum(g * x)
        hvp = self._flat_concat(tape1.gradient(l, pi_params))
        if DAMPING_COEFF > 0:
            hvp += DAMPING_COEFF * x
        return hvp.numpy()

    def cg(self, Ax, b):
        x = np.zeros_like(b)
        r = copy.deepcopy(b)
        p = copy.deepcopy(r)
        r_dot_old = np.dot(r, r)
        for _ in range(CG_ITERS):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def update(self):
        states, actions, adv, rewards_to_go, old_logp, old_mu, old_log_std = self.buf.get()
        g, old_loss = self.gradient(states, actions, adv, old_logp)
        Hx = lambda v: self.hvp(states, old_mu, old_log_std, v)
        x = self.cg(Hx, g)
        alpha = np.sqrt(2 * DELTA / (np.dot(x, Hx(x)) + EPS))
        old_params = self.get_pi_params()

        # Backtracking line search
        for j in range(BACKTRACK_ITERS):
            step = BACKTRACK_COEFF**j
            self.set_pi_params(old_params - alpha * x * step)
            kl = self.kl(states, old_mu, old_log_std)
            new_loss = self.pi_loss(states, actions, adv, old_logp)
            if kl <= DELTA and new_loss <= old_loss:
                break
        else:
            self.set_pi_params(old_params)  # restore if line search fails

        # Train value function
        self.train_vf(states, rewards_to_go)

    def finish_path(self, done, next_state):
        last_val = 0 if done else self.critic(np.array([next_state], np.float32))[0, 0]
        self.buf.finish_path(last_val)


    def save(self):  # save trained weights
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_npz_dict(self.actor.trainable_weights, name=os.path.join(path, 'model_actor.npz'))
        tl.files.save_npz_dict(self.critic.trainable_weights, name=os.path.join(path, 'model_critic.npz'))

    def load(self):  # load trained weights
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_and_assign_npz_dict(name=os.path.join(path, 'model_actor.npz'), network=self.actor)
        tl.files.load_and_assign_npz_dict(name=os.path.join(path, 'model_critic.npz'), network=self.critic)


##################### Main ####################
if __name__ == '__main__':
    env = gym.make(ENV_ID)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    env.action_space.seed(RANDOM_SEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    agent = TRPO(state_dim, action_dim, action_bound)
    t0 = time.time()

    if args.train:
        all_episode_reward = []
        for episode in range(TRAIN_EPISODES):
            state, _ = env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                if RENDER: env.render()
                action, value, logp, mean, log_std = agent.get_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.buf.store(state, action, reward, value, logp, mean, log_std)
                episode_reward += reward
                state = next_state
                if agent.buf.is_full():
                    agent.finish_path(done, next_state)
                    agent.update()
                if done: break
            agent.finish_path(done, next_state)
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            print(f'Training | Episode {episode+1}/{TRAIN_EPISODES} | Reward: {episode_reward:.2f} | Time: {time.time()-t0:.2f}s')
            if episode % SAVE_FREQ == 0:
                agent.save()
        agent.save()
        plt.plot(all_episode_reward)
        os.makedirs('image', exist_ok=True)
        plt.savefig(os.path.join('image', f'{ALG_NAME}_{ENV_ID}.png'))

    if args.test:
        agent.load()
        for episode in range(TEST_EPISODES):
            state, _ = env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                env.render()
                action, *_ = agent.get_action(state, greedy=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                state = next_state
                if done: break
            print(f'Testing  | Episode {episode+1}/{TEST_EPISODES} | Reward: {episode_reward:.2f} | Time: {time.time()-t0:.2f}s')
