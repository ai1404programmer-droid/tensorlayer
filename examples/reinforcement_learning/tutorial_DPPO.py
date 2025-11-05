"""
Distributed Proximal Policy Optimization (DPPO)
-----------------------------------------------
Distributed version of PPO with multiple workers for continuous action space.

Environment
-----------
Gymnasium Pendulum-v1

"""

import argparse
import os
import queue
import threading
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test DPPO agent.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

# ---------------- Hyperparameters ----------------
ENV_ID = 'Pendulum-v1'
RANDOMSEED = 2
RENDER = False

ALG_NAME = 'DPPO'
TRAIN_EPISODES = 10
TEST_EPISODES = 10
MAX_STEPS = 10
GAMMA = 0.9
LR_A = 0.0001
LR_C = 0.0002
ACTOR_UPDATE_STEPS = 10
CRITIC_UPDATE_STEPS = 10
MIN_BATCH_SIZE = 64

N_WORKER = 4
UPDATE_STEP = 10

# PPO clip parameters
EPSILON = 0.2

# ---------------- DPPO Class ----------------
class PPO:
    def __init__(self, state_dim, action_dim, action_bound, method='clip'):
        # Critic
        inputs = tl.layers.Input([None, state_dim], tf.float32)
        x = tl.layers.Dense(64, tf.nn.relu)(inputs)
        x = tl.layers.Dense(64, tf.nn.relu)(x)
        v = tl.layers.Dense(1)(x)
        self.critic = tl.models.Model(inputs, v)
        self.critic.train()

        # Actor
        inputs = tl.layers.Input([None, state_dim], tf.float32)
        x = tl.layers.Dense(64, tf.nn.relu)(inputs)
        x = tl.layers.Dense(64, tf.nn.relu)(x)
        a = tl.layers.Dense(action_dim, tf.nn.tanh)(x)
        mean = tl.layers.Lambda(lambda x: x * action_bound)(a)
        logstd = tf.Variable(np.zeros(action_dim, dtype=np.float32))
        self.actor = tl.models.Model(inputs, mean)
        self.actor.trainable_weights.append(logstd)
        self.actor.logstd = logstd
        self.actor.train()

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

        self.method = method
        if method == 'clip':
            self.epsilon = EPSILON

        self.state_buffer, self.action_buffer = [], []
        self.reward_buffer = []
        self.action_bound = action_bound

    def train_actor(self, state, action, adv, old_pi):
        with tf.GradientTape() as tape:
            mean, std = self.actor(state), tf.exp(self.actor.logstd)
            pi = tfp.distributions.Normal(mean, std)
            ratio = tf.exp(pi.log_prob(action) - old_pi.log_prob(action))
            surr = ratio * adv
            loss = -tf.reduce_mean(
                tf.minimum(surr, tf.clip_by_value(ratio, 1. - self.epsilon, 1. + self.epsilon) * adv)
            )
        grads = tape.gradient(loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_weights))

    def train_critic(self, reward, state):
        reward = np.array(reward, dtype=np.float32)
        with tf.GradientTape() as tape:
            advantage = reward - self.critic(state)
            loss = tf.reduce_mean(tf.square(advantage))
        grads = tape.gradient(loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_weights))

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < TRAIN_EPISODES:
                UPDATE_EVENT.wait()
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]
                s, a, r = zip(*data)
                s = np.vstack(s).astype(np.float32)
                a = np.vstack(a).astype(np.float32)
                r = np.vstack(r).astype(np.float32)
                mean, std = self.actor(s), tf.exp(self.actor.logstd)
                pi = tfp.distributions.Normal(mean, std)
                adv = r - self.critic(s)

                for _ in range(ACTOR_UPDATE_STEPS):
                    self.train_actor(s, a, adv, pi)
                for _ in range(CRITIC_UPDATE_STEPS):
                    self.train_critic(r, s)

                UPDATE_EVENT.clear()
                GLOBAL_UPDATE_COUNTER = 0
                ROLLING_EVENT.set()

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

        tl.files.save_npz_dict(self.actor.trainable_weights, name=os.path.join(path, 'model_actor.npz'))
        tl.files.save_npz_dict(self.critic.trainable_weights, name=os.path.join(path, 'model_critic.npz'))
    def load(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))

        tl.files.load_and_assign_npz_dict(name=os.path.join(path, 'model_actor.npz'), network=self.actor)
        tl.files.load_and_assign_npz_dict(name=os.path.join(path, 'model_critic.npz'), network=self.critic)

# ---------------- Worker Class ----------------
class Worker:
    def __init__(self, wid):
        self.wid = wid
        self.env = gym.make(ENV_ID)
        self.env.reset(seed=RANDOMSEED + wid)
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            s, _ = self.env.reset(seed=RANDOMSEED + self.wid)
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(MAX_STEPS):
                if not ROLLING_EVENT.is_set():
                    ROLLING_EVENT.wait()
                    buffer_s, buffer_a, buffer_r = [], [], []

                a = self.ppo.get_action(s)
                s_, r, terminated, truncated, info = self.env.step(a)
                done = terminated or truncated
                if RENDER and self.wid == 0:
                    self.env.render()

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1
                if t == MAX_STEPS - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    v_s_ = 0 if done else self.ppo.critic(np.array([s_], np.float32))[0][0]
                    discounted_r = []
                    for reward in buffer_r[::-1]:
                        v_s_ = reward + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()
                    buffer_r = np.array(discounted_r)[:, np.newaxis]
                    QUEUE.put([buffer_s, buffer_a, buffer_r])
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()
                        UPDATE_EVENT.set()

                    if GLOBAL_EP >= TRAIN_EPISODES:
                        COORD.request_stop()
                        break

            print(f'Training | Episode: {GLOBAL_EP+1}/{TRAIN_EPISODES} | Worker: {self.wid} | Reward: {ep_r:.4f} | Time: {time.time()-T0:.2f}')
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1

# ---------------- Main ----------------
if __name__ == '__main__':
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    env = gym.make(ENV_ID)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high
    env.close()

    GLOBAL_PPO = PPO(state_dim, action_dim, action_bound)
    T0 = time.time()

    if args.train:
        UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
        UPDATE_EVENT.clear()
        ROLLING_EVENT.set()
        workers = [Worker(i) for i in range(N_WORKER)]

        GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
        GLOBAL_RUNNING_R = []
        COORD = tf.train.Coordinator()
        QUEUE = queue.Queue()

        threads = []
        for worker in workers:
            t = threading.Thread(target=worker.work)
            t.start()
            threads.append(t)

        threads.append(threading.Thread(target=GLOBAL_PPO.update))
        threads[-1].start()
        COORD.join(threads)

        GLOBAL_PPO.save()
        plt.plot(GLOBAL_RUNNING_R)
        os.makedirs('image', exist_ok=True)
        plt.savefig(os.path.join('image', f'{ALG_NAME}_{ENV_ID}.png'))

    if args.test:
        GLOBAL_PPO.load()
        env = gym.make(ENV_ID)
        for episode in range(TEST_EPISODES):
            state, _ = env.reset(seed=RANDOMSEED)
            ep_r = 0
            for step in range(MAX_STEPS):
                env.render()
                s_, r, terminated, truncated, info = env.step(GLOBAL_PPO.get_action(state, greedy=True))
                done = terminated or truncated
                ep_r += r
                state = s_
                if done:
                    break
            print(f'Testing | Episode: {episode+1}/{TEST_EPISODES} | Reward: {ep_r:.4f} | Time: {time.time()-T0:.2f}')
        env.close()
