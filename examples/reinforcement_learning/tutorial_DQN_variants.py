"""
DQN and its variants with npz save/load
---------------------------------------
Supports: Double DQN, Dueling DQN, Noisy DQN
Environment: CartPole-v0 or PongNoFrameskip-v4
Requirements: tensorflow>=2.0.0a0, tensorlayer>=2.0.0
Run:
python tutorial_DQN_variants.py --train
python tutorial_DQN_variants.py --test --save_path=dqn_variants/
"""
"""
DQN and its variants (Double, Dueling, Noisy)
Compatible with gymnasium (5 return values)
"""
import argparse
import os
import random
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl

# ---------------- Arguments ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)
parser.add_argument('--save_path', default=None, help='folder to save/load model')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--env_id', default='CartPole-v0')
parser.add_argument('--noisy_scale', type=float, default=1e-2)
parser.add_argument('--disable_double', action='store_true', default=False)
parser.add_argument('--disable_dueling', action='store_true', default=False)
args = parser.parse_args()

# ---------------- Seed ----------------
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# ---------------- Environment ----------------
env_id = args.env_id
env = gym.make(env_id)
env.reset(seed=args.seed)
noise_scale = args.noisy_scale
double = not args.disable_double
dueling = not args.disable_dueling

alg_name = 'DQN'
if dueling: alg_name = 'Dueling_' + alg_name
if double: alg_name = 'Double_' + alg_name
if noise_scale != 0: alg_name = 'Noisy_' + alg_name
print(alg_name)

# ---------------- Hyperparameters ----------------
if env_id == 'CartPole-v0':
    qnet_type = 'MLP'
    number_timesteps = 10
    explore_timesteps = 10
    epsilon = lambda i_iter: 1 - 0.99 * min(1, i_iter / explore_timesteps)
    lr = 5e-3
    buffer_size = 1000
    target_q_update_freq = 50
    ob_scale = 1.0
    clipnorm = None
else:
    qnet_type = 'CNN'
    number_timesteps = int(1e6)
    explore_timesteps = 1e5
    epsilon = lambda i_iter: 1 - 0.99 * min(1, i_iter / explore_timesteps)
    lr = 1e-4
    buffer_size = 10000
    target_q_update_freq = 200
    ob_scale = 1.0 / 255
    clipnorm = 10

in_dim = env.observation_space.shape
out_dim = env.action_space.n
reward_gamma = 0.99
batch_size = 32
warm_start = buffer_size / 10
noise_update_freq = 50

# ---------------- Networks ----------------
class MLP(tl.models.Model):
    def __init__(self, name):
        super(MLP, self).__init__(name=name)
        self.h1 = tl.layers.Dense(64, tf.nn.tanh, in_channels=in_dim[0])
        self.qvalue = tl.layers.Dense(out_dim, in_channels=64, name='q', W_init=tf.initializers.GlorotUniform())
        self.svalue = tl.layers.Dense(1, in_channels=64, name='s', W_init=tf.initializers.GlorotUniform())
        self.noise_scale = 0

    def forward(self, ni):
        feature = self.h1(ni)
        if self.noise_scale != 0:
            noises = []
            for layer in [self.qvalue, self.svalue]:
                for var in layer.trainable_weights:
                    noise = tf.random.normal(tf.shape(var), 0, self.noise_scale)
                    noises.append(noise)
                    var.assign_add(noise)
        qvalue = self.qvalue(feature)
        svalue = self.svalue(feature)
        if self.noise_scale != 0:
            idx = 0
            for layer in [self.qvalue, self.svalue]:
                for var in layer.trainable_weights:
                    var.assign_sub(noises[idx])
                    idx += 1
        if dueling:
            return svalue + qvalue - tf.reduce_mean(qvalue, 1, keepdims=True)
        else:
            return qvalue

# ---------------- Replay Buffer ----------------
class ReplayBuffer:
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, *args):
        if self._next_idx >= len(self._storage):
            self._storage.append(args)
        else:
            self._storage[self._next_idx] = args
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        b_o, b_a, b_r, b_o_, b_d = [], [], [], [], []
        for i in idxes:
            o, a, r, o_, d = self._storage[i]
            b_o.append(o)
            b_a.append(a)
            b_r.append(r)
            b_o_.append(o_)
            b_d.append(d)
        return (
            np.stack(b_o).astype('float32') * ob_scale,
            np.stack(b_a).astype('int32'),
            np.stack(b_r).astype('float32'),
            np.stack(b_o_).astype('float32') * ob_scale,
            np.stack(b_d).astype('float32'),
        )

    def sample(self, batch_size):
        idxes = [random.choice(range(len(self._storage))) for _ in range(batch_size)]
        return self._encode_sample(idxes)

# ---------------- Utility ----------------
def huber_loss(x):
    return tf.where(tf.abs(x) < 1, tf.square(x) * 0.5, tf.abs(x) - 0.5)

def sync(net, net_tar):
    for var, var_tar in zip(net.trainable_weights, net_tar.trainable_weights):
        var_tar.assign(var)

def log_softmax(x, dim):
    temp = x - np.max(x, dim, keepdims=True)
    return temp - np.log(np.exp(temp).sum(dim, keepdims=True))

def softmax(x, dim):
    temp = np.exp(x - np.max(x, dim, keepdims=True))
    return temp / temp.sum(dim, keepdims=True)

# ---------------- DQN Agent ----------------
class DQN:
    def __init__(self):
        model = MLP if qnet_type == 'MLP' else CNN
        self.qnet = model('q')
        if args.train:
            self.qnet.train()
            self.targetqnet = model('targetq')
            self.targetqnet.infer()
            sync(self.qnet, self.targetqnet)
        else:
            self.qnet.infer()
            self.load(args.save_path)
        self.niter = 0
        self.noise_scale = noise_scale
        self.optimizer = tf.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm if clipnorm else None)

    def get_action(self, obv):
        eps = epsilon(self.niter)
        obv = np.expand_dims(obv, 0).astype('float32') * ob_scale
        if args.train and random.random() < eps:
            return int(random.random() * out_dim)
        self.qnet.noise_scale = self.noise_scale if self.niter < explore_timesteps else 0
        q = self.qnet(obv).numpy()
        self.qnet.noise_scale = 0
        return q.argmax(1)[0]

    @tf.function
    def _qvalues_func(self, obv):
        return self.qnet(obv)

    def train(self, b_o, b_a, b_r, b_o_, b_d):
        self._train_func(b_o, b_a, b_r, b_o_, b_d)
        self.niter += 1
        if self.niter % target_q_update_freq == 0:
            sync(self.qnet, self.targetqnet)
            self.save(args.save_path)

    @tf.function
    def _train_func(self, b_o, b_a, b_r, b_o_, b_d):
        with tf.GradientTape() as tape:
            td_errors = self._tderror_func(b_o, b_a, b_r, b_o_, b_d)
            loss = tf.reduce_mean(huber_loss(td_errors))
        grad = tape.gradient(loss, self.qnet.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.qnet.trainable_weights))
        return td_errors

    @tf.function
    def _tderror_func(self, b_o, b_a, b_r, b_o_, b_d):
        if double:
            b_a_ = tf.one_hot(tf.argmax(self.qnet(b_o_), 1), out_dim)
            b_q_ = (1 - b_d) * tf.reduce_sum(self.targetqnet(b_o_) * b_a_, 1)
        else:
            b_q_ = (1 - b_d) * tf.reduce_max(self.targetqnet(b_o_), 1)
        b_q = tf.reduce_sum(self.qnet(b_o) * tf.one_hot(b_a, out_dim), 1)
        return b_q - (b_r + reward_gamma * b_q_)

    def save(self, path):
        if path is None:
            path = os.path.join('model', '_'.join([alg_name, env_id]))
        os.makedirs(path, exist_ok=True)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'q_net.hdf5'), self.qnet)

    def load(self, path):
        if path is None:
            path = os.path.join('model', '_'.join([alg_name, env_id]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'q_net.hdf5'), self.qnet)

# ---------------- Training & Testing ----------------
if __name__ == '__main__':
    dqn = DQN()
    t0 = time.time()
    if args.train:
        buffer = ReplayBuffer(buffer_size)
        nepisode = 0
        all_episode_reward = []
        for i in range(1, number_timesteps + 1):
            o, _ = env.reset()
            episode_reward = 0
            while True:
                a = dqn.get_action(o)
                o_, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                buffer.add(o, a, r, o_, done)
                episode_reward += r

                if i >= warm_start:
                    transitions = buffer.sample(batch_size)
                    dqn.train(*transitions)

                if done:
                    break
                else:
                    o = o_

            if nepisode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            nepisode += 1
            print(f'Training  | Episode: {nepisode}  | Reward: {episode_reward:.4f}  | Time: {time.time()-t0:.2f}')

        dqn.save(args.save_path)
        plt.plot(all_episode_reward)
        os.makedirs('image', exist_ok=True)
        plt.savefig(os.path.join('image', '_'.join([alg_name, env_id])))

    if args.test:
        nepisode = 0
        for i in range(1, number_timesteps + 1):
            o, _ = env.reset()
            episode_reward = 0
            while True:
                env.render()
                a = dqn.get_action(o)
                o_, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                episode_reward += r
                if done:
                    break
                else:
                    o = o_
            nepisode += 1
            print(f'Testing  | Episode: {nepisode}  | Reward: {episode_reward:.4f}  | Time: {time.time()-t0:.2f}')
