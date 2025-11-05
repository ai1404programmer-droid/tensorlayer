"""
Prioritized Experience Replay (PER) - TensorLayer + TensorFlow 2.x
-----------------------------------------------------------------
سازگار با:
- TensorFlow >= 2.0
- TensorLayer >= 2.2
- Gymnasium (نسخه جدید gym)
تغییرات:
- env.step() → observation, reward, terminated, truncated, info
- env.reset() → observation, info
- محاسبه done = terminated or truncated
"""

import argparse
import operator
import os
import random
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl

# ====================== Argument Parser ======================
parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)
parser.add_argument('--save_path', default=None, help='path to save/load model')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--env_id', default='CartPole-v1', help='CartPole-v1 or PongNoFrameskip-v4')
args = parser.parse_args()

# ====================== Seeds ======================
random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# ====================== Environment ======================
env_id = args.env_id
env = gym.make(env_id)
env.reset(seed=args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)

alg_name = 'prioritized_replay'

# ====================== Hyperparameters ======================
if env_id.startswith('CartPole'):
    qnet_type = 'MLP'
    number_timesteps = 10
    explore_timesteps = 10
    epsilon = lambda i: 1 - 0.99 * min(1, i / explore_timesteps)
    lr = 5e-3
    buffer_size = 1000
    target_q_update_freq = 50
    ob_scale = 1.0
    clipnorm = None
else:
    qnet_type = 'CNN'
    number_timesteps = int(1e6)
    explore_timesteps = 1e5
    epsilon = lambda i: 1 - 0.99 * min(1, i / explore_timesteps)
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
prioritized_replay_alpha = 0.6
prioritized_replay_beta0 = 0.4

# ====================== Network ======================
class MLP(tl.models.Model):
    def __init__(self, name='MLP'):
        super(MLP, self).__init__(name=name)
        self.h1 = tl.layers.Dense(64, tf.nn.tanh, in_channels=in_dim[0])
        self.qvalue = tl.layers.Dense(out_dim, in_channels=64, name='q',
                                      W_init=tf.initializers.GlorotUniform())
    def forward(self, ni):
        return self.qvalue(self.h1(ni))

class CNN(tl.models.Model):
    def __init__(self, name='CNN'):
        super(CNN, self).__init__(name=name)
        h, w, in_channels = in_dim
        dense_in_channels = 64 * ((h - 28) // 8) * ((w - 28) // 8)
        self.conv1 = tl.layers.Conv2d(32, (8,8), (4,4), tf.nn.relu, 'VALID', in_channels=in_channels)
        self.conv2 = tl.layers.Conv2d(64, (4,4), (2,2), tf.nn.relu, 'VALID', in_channels=32)
        self.conv3 = tl.layers.Conv2d(64, (3,3), (1,1), tf.nn.relu, 'VALID', in_channels=64)
        self.flatten = tl.layers.Flatten()
        self.preq = tl.layers.Dense(256, tf.nn.relu, in_channels=dense_in_channels)
        self.qvalue = tl.layers.Dense(out_dim, in_channels=256)
    def forward(self, ni):
        feature = self.flatten(self.conv3(self.conv2(self.conv1(ni))))
        return self.qvalue(self.preq(feature))

# ====================== Replay Buffers ======================
class SegmentTree:
    def __init__(self, capacity, operation, neutral_element):
        assert capacity > 0 and capacity & (capacity - 1) == 0
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2*capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2*node, node_start, mid)
        elif mid+1 <= start:
            return self._reduce_helper(start, end, 2*node+1, mid+1, node_end)
        else:
            return self._operation(
                self._reduce_helper(start, mid, 2*node, node_start, mid),
                self._reduce_helper(mid+1, end, 2*node+1, mid+1, node_end)
            )

    def reduce(self, start=0, end=None):
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity-1)

    def __setitem__(self, idx, val):
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(self._value[2*idx], self._value[2*idx+1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]

class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, operator.add, 0.0)
    def sum(self, start=0, end=None):
        return super().reduce(start, end)
    def find_prefixsum_idx(self, prefixsum):
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:
            if self._value[2*idx] > prefixsum:
                idx = 2*idx
            else:
                prefixsum -= self._value[2*idx]
                idx = 2*idx + 1
        return idx - self._capacity

class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super().__init__(capacity, min, float('inf'))
    def min(self, start=0, end=None):
        return super().reduce(start, end)

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
        return (np.stack(b_o).astype('float32') * ob_scale,
                np.stack(b_a).astype('int32'),
                np.stack(b_r).astype('float32'),
                np.stack(b_o_).astype('float32') * ob_scale,
                np.stack(b_d).astype('float32'))
    def sample(self, batch_size):
        indexes = range(len(self._storage))
        idxes = [random.choice(indexes) for _ in range(batch_size)]
        return self._encode_sample(idxes)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, beta):
        super().__init__(size)
        self._alpha = alpha
        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2
        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self.beta = beta
    def add(self, *args):
        idx = self._next_idx
        super().add(*args)
        self._it_sum[idx] = self._max_priority**self._alpha
        self._it_min[idx] = self._max_priority**self._alpha
    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self._storage)-1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res
    def sample(self, batch_size):
        idxes = self._sample_proportional(batch_size)
        it_sum = self._it_sum.sum()
        p_min = self._it_min.min() / it_sum
        max_weight = (p_min * len(self._storage))**(-self.beta)
        p_samples = np.asarray([self._it_sum[idx] for idx in idxes]) / it_sum
        weights = (p_samples * len(self._storage))**(-self.beta) / max_weight
        encoded_sample = self._encode_sample(idxes)
        return encoded_sample + (weights.astype('float32'), idxes)
    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority**self._alpha
            self._it_min[idx] = priority**self._alpha
            self._max_priority = max(self._max_priority, priority)

# ====================== Helper Functions ======================
def huber_loss(x):
    return tf.where(tf.abs(x)<1, 0.5*tf.square(x), tf.abs(x)-0.5)

def sync(net, net_tar):
    for var, var_tar in zip(net.trainable_weights, net_tar.trainable_weights):
        var_tar.assign(var)

# ====================== DQN ======================
class DQN:
    def __init__(self):
        model = MLP if qnet_type=='MLP' else CNN
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
        if clipnorm is not None:
            self.optimizer = tf.optimizers.Adam(lr, clipnorm=clipnorm)
        else:
            self.optimizer = tf.optimizers.Adam(lr)
    def get_action(self, obv):
        eps = epsilon(self.niter)
        if args.train and random.random() < eps:
            return int(random.random()*out_dim)
        else:
            obv = np.expand_dims(obv,0).astype('float32')*ob_scale
            return self._qvalues_func(obv).numpy().argmax(1)[0]
    @tf.function
    def _qvalues_func(self, obv):
        return self.qnet(obv)
    def train(self, b_o, b_a, b_r, b_o_, b_d, weights=None):
        if weights is None:
            weights = np.ones_like(b_r)
        td_errors = self._train_func(b_o, b_a, b_r, b_o_, b_d, weights)
        self.niter += 1
        if self.niter % target_q_update_freq == 0:
            sync(self.qnet, self.targetqnet)
            self.save(args.save_path)
        return td_errors.numpy()
    def save(self, path):
        if path is None:
            path = os.path.join('model','_'.join([alg_name, env_id]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_weights_to_hdf5(os.path.join(path,'q_net.hdf5'), self.qnet)
    def load(self, path):
        if path is None:
            path = os.path.join('model','_'.join([alg_name, env_id]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path,'q_net.hdf5'), self.qnet)
    @tf.function
    def _train_func(self, b_o, b_a, b_r, b_o_, b_d, weights):
        with tf.GradientTape() as tape:
            td_errors = self._tderror_func(b_o, b_a, b_r, b_o_, b_d)
            loss = tf.reduce_mean(huber_loss(td_errors) * weights)
        grad = tape.gradient(loss, self.qnet.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.qnet.trainable_weights))
        return td_errors
    @tf.function
    def _tderror_func(self, b_o, b_a, b_r, b_o_, b_d):
        b_q_ = (1-b_d) * tf.reduce_max(self.targetqnet(b_o_),1)
        b_q = tf.reduce_sum(self.qnet(b_o) * tf.one_hot(b_a, out_dim),1)
        return b_q - (b_r + reward_gamma * b_q_)

# ====================== Training / Testing ======================
if __name__ == '__main__':
    dqn = DQN()
    t0 = time.time()

    if args.train:
        buffer = PrioritizedReplayBuffer(buffer_size, prioritized_replay_alpha, prioritized_replay_beta0)
        nepisode = 0
        all_episode_reward = []

        for i in range(1, number_timesteps+1):
            o, _ = env.reset()
            episode_reward = 0
            while True:
                buffer.beta += (1 - prioritized_replay_beta0) / number_timesteps
                a = dqn.get_action(o)

                # Gymnasium new API
                o_, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated

                buffer.add(o, a, r, o_, done)
                episode_reward += r

                if i >= warm_start:
                    *transitions, idxs = buffer.sample(batch_size)
                    priorities = dqn.train(*transitions)
                    priorities = np.clip(np.abs(priorities), 1e-6, None)
                    buffer.update_priorities(idxs, priorities)

                if done:
                    break
                else:
                    o = o_

            if nepisode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1]*0.9 + episode_reward*0.1)
            nepisode += 1
            print(f'Training  | Episode: {nepisode}  | Reward: {episode_reward:.4f}  | Time: {time.time()-t0:.4f}')

        dqn.save(args.save_path)
        plt.plot(all_episode_reward)
        os.makedirs('image', exist_ok=True)
        plt.savefig(os.path.join('image','_'.join([alg_name, env_id])))

    if args.test:
        nepisode = 0
        for i in range(1, number_timesteps+1):
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
            print(f'Testing  | Episode: {nepisode}  | Reward: {episode_reward:.4f}  | Time: {time.time()-t0:.4f}')
