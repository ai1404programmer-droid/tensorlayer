"""
C51 Algorithm
------------------------
Categorical 51 distributional RL algorithm, 51 means the number of atoms. In
this algorithm, instead of estimating actual expected value, value distribution
over a series of continuous sub-intervals (atoms) is considered.
Reference:
------------------------
Bellemare M G, Dabney W, Munos R. A distributional perspective on reinforcement
learning[C]//Proceedings of the 34th International Conference on Machine
Learning-Volume 70. JMLR. org, 2017: 449-458.
Environment:
------------------------
Cartpole and Pong in OpenAI Gym
Requirements:
------------------------
tensorflow>=2.0.0a0
tensorlayer>=2.0.0
To run:
------------------------
python tutorial_C51.py --mode=train
python tutorial_C51.py --mode=test --save_path=c51/8000.npz"""

import argparse
import os
import random
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl

# ------------------------ Arguments ------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)
parser.add_argument('--save_path', default=None,
                    help='Folder to save if train else path to load model')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--env_id', default='CartPole-v0', help='CartPole-v0 or PongNoFrameskip-v4')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

# ------------------------ Environment ------------------------
env_id = args.env_id
env = gym.make(env_id)
alg_name = 'C51'

# ------------------------ Hyperparameters ------------------------
if env_id == 'CartPole-v0':
    qnet_type = 'MLP'
    number_timesteps = 10
    explore_timesteps = 100
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
warm_start = buffer_size // 10
atom_num = 51
min_value = -10
max_value = 10
vrange = np.linspace(min_value, max_value, atom_num)
deltaz = float(max_value - min_value) / (atom_num - 1)

# ------------------------ Networks ------------------------
class MLP(tl.models.Model):
    def __init__(self, name):
        super().__init__(name=name)
        self.h1 = tl.layers.Dense(64, tf.nn.tanh, in_channels=in_dim[0])
        self.qvalue = tl.layers.Dense(out_dim * atom_num, in_channels=64)
        self.reshape = tl.layers.Reshape((-1, out_dim, atom_num))

    def forward(self, ni):
        qvalues = self.qvalue(self.h1(ni))
        return tf.nn.log_softmax(self.reshape(qvalues), axis=2)

# CNN omitted for brevity (CartPole معمولاً MLP کافیست)

# ------------------------ Replay Buffer ------------------------
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
            np.stack(b_d).astype('float32')
        )

    def sample(self, batch_size):
        idxes = [random.choice(range(len(self._storage))) for _ in range(batch_size)]
        return self._encode_sample(idxes)

# ------------------------ Utilities ------------------------
def sync(net, net_tar):
    for var, var_tar in zip(net.trainable_weights, net_tar.trainable_weights):
        var_tar.assign(var)

# ------------------------ C51 DQN ------------------------
class DQN:
    def __init__(self):
        model_cls = MLP if qnet_type == 'MLP' else CNN
        self.qnet = model_cls('q')
        if args.train:
            self.qnet.train()
            self.targetqnet = model_cls('targetq')
            self.targetqnet.infer()
            sync(self.qnet, self.targetqnet)
        else:
            self.qnet.infer()
            self.load(args.save_path)

        self.niter = 0
        self.optimizer = tf.optimizers.Adam(learning_rate=lr, clipnorm=clipnorm)

    def get_action(self, obv):
        eps = epsilon(self.niter)
        if args.train and random.random() < eps:
            return random.randint(0, out_dim - 1)
        obv = np.expand_dims(obv, 0).astype('float32') * ob_scale
        qdist = np.exp(self.qnet(obv).numpy())
        qvalues = (qdist * vrange).sum(-1)
        return qvalues.argmax(1)[0]

    @tf.function
    def _train_func(self, b_o, b_index, b_m):
        with tf.GradientTape() as tape:
            b_dist_a = tf.gather_nd(self.qnet(b_o), b_index)
            loss = tf.reduce_mean(-tf.reduce_sum(b_dist_a * b_m, axis=1))
        grad = tape.gradient(loss, self.qnet.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.qnet.trainable_weights))

    def train(self, b_o, b_a, b_r, b_o_, b_d):
        b_dist_ = np.exp(self.targetqnet(b_o_).numpy())
        b_a_ = (b_dist_ * vrange).sum(-1).argmax(1)
        b_tzj = np.clip(reward_gamma * (1 - b_d[:, None]) * vrange[None, :] + b_r[:, None], min_value, max_value)
        b_i = (b_tzj - min_value) / deltaz
        b_l = np.floor(b_i).astype('int64')
        b_u = np.ceil(b_i).astype('int64')
        templ = b_dist_[range(batch_size), b_a_, :] * (b_u - b_i)
        tempu = b_dist_[range(batch_size), b_a_, :] * (b_i - b_l)
        b_m = np.zeros((batch_size, atom_num))
        for j in range(batch_size):
            for k in range(atom_num):
                b_m[j][b_l[j][k]] += templ[j][k]
                b_m[j][b_u[j][k]] += tempu[j][k]
        b_m = tf.convert_to_tensor(b_m, dtype='float32')
        b_index = tf.convert_to_tensor(np.stack([range(batch_size), b_a], 1), 'int64')
        self._train_func(b_o, b_index, b_m)

        self.niter += 1
        if self.niter % target_q_update_freq == 0:
            sync(self.qnet, self.targetqnet)
            self.save(args.save_path)

    def save(self, path):
        if path is None:
            path = os.path.join('model', '_'.join([alg_name, env_id]))
        os.makedirs(path, exist_ok=True)
        tl.files.save_weights_to_hdf5(os.path.join(path, 'q_net.hdf5'), self.qnet)

    def load(self, path):
        if path is None:
            path = os.path.join('model', '_'.join([alg_name, env_id]))
        tl.files.load_hdf5_to_weights_in_order(os.path.join(path, 'q_net.hdf5'), self.qnet)

# ------------------------ Training ------------------------
if __name__ == '__main__':
    dqn = DQN()
    t0 = time.time()
    if args.train:
        buffer = ReplayBuffer(buffer_size)
        all_episode_reward = []
        for i in range(1, number_timesteps + 1):
            o, info = env.reset(seed=args.seed)
            episode_reward = 0
            while True:
                a = dqn.get_action(o)
                o_, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                buffer.add(o, a, r, o_, done)
                episode_reward += r
                if i >= warm_start and len(buffer) >= batch_size:
                    transitions = buffer.sample(batch_size)
                    dqn.train(*transitions)
                o = o_
                if done:
                    break
            all_episode_reward.append(episode_reward if i == 1 else all_episode_reward[-1] * 0.9 + episode_reward * 0.1)
            print(f'Training | Step: {i} | Episode Reward: {episode_reward:.4f} | Running Time: {time.time()-t0:.2f}')
        dqn.save(args.save_path)
        plt.plot(all_episode_reward)
        os.makedirs('image', exist_ok=True)
        plt.savefig(os.path.join('image', f'{alg_name}_{env_id}.png'))

    # ------------------------ Testing ------------------------
    if args.test:
        for i in range(10):
            o, info = env.reset(seed=args.seed)
            episode_reward = 0
            while True:
                env.render()
                a = dqn.get_action(o)
                o_, r, terminated, truncated, info = env.step(a)
                done = terminated or truncated
                episode_reward += r
                o = o_
                if done:
                    break
            print(f'Testing | Episode: {i+1} | Reward: {episode_reward:.4f} | Running Time: {time.time()-t0:.2f}')
