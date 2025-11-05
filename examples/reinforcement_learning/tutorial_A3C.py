"""
Asynchronous Advantage Actor Critic (A3C) with Continuous Action Space.

Actor Critic History
----------------------
A3C > DDPG (for continuous action space) > AC

Advantage
----------
Train faster and more stable than AC.

Disadvantage
-------------
Have bias.

Reference
----------
Original Paper: https://arxiv.org/pdf/1602.01783.pdf
MorvanZhou's tutorial: https://morvanzhou.github.io/tutorials/
MorvanZhou's code: https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/experiments/Solve_BipedalWalker/A3C.py

Environment
-----------
BipedalWalker-v2 : https://gym.openai.com/envs/BipedalWalker-v2

Reward is given for moving forward, total 300+ points up to the far end.
If the robot falls, it gets -100. Applying motor torque costs a small amount of
points, more optimal agent will get better score. State consists of hull angle
speed, angular velocity, horizontal speed, vertical speed, position of joints
and joints angular speed, legs contact with ground, and 10 lidar rangefinder
measurements. There's no coordinates in the state vector.

Prerequisites
--------------
tensorflow 2.0.0a0
tensorflow-probability 0.6.0
tensorlayer 2.0.0
&&
pip install box2d box2d-kengz --user

To run
------
python tutorial_A3C.py --train/test

"""

import threading
import time
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import tensorflow as tf
import tensorflow_probability as tfp
import tensorlayer as tl
from tensorlayer.layers import Dense
import multiprocessing
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # فقط خطاهای مهم رو نمایش بده

tfd = tfp.distributions

tl.logging.set_verbosity(tl.logging.DEBUG)

# add arguments in command  --train/test
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

#####################  hyper parameters  ####################

ENV_ID = 'BipedalWalker-v3'  #CartPole-v1  BipedalWalkerHardcore-v2   BipedalWalker-v2  LunarLanderContinuous-v2
RANDOM_SEED = 2  # random seed, can be either an int number or None
RENDER = False  # render while training

ALG_NAME = 'A3C'
N_WORKERS = multiprocessing.cpu_count()  # number of workers according to number of cores in cpu
# N_WORKERS = 2     # manually set number of workers
MAX_GLOBAL_EP = 5 #15000  # number of training episodes
TEST_EPISODES = 10  # number of training episodes
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10  # update global policy after several episodes
GAMMA = 0.99  # reward discount factor
ENTROPY_BETA = 0.005  # factor for entropy boosted exploration
LR_A = 0.00005  # learning rate for actor
LR_C = 0.0001  # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0  # will increase during training, stop training when it >= MAX_GLOBAL_EP

###################  Asynchronous Advantage Actor Critic (A3C)  ####################################


class ACNet(object):

    def __init__(self, scope):
        self.scope = scope

        w_init = tf.keras.initializers.glorot_normal(seed=None)  # initializer, glorot=xavier

        def get_actor(input_shape):  # policy network
            with tf.name_scope(self.scope):
                ni = tl.layers.Input(input_shape, name='in')
                nn = tl.layers.Dense(n_units=500, act=tf.nn.relu6, W_init=w_init, name='la')(ni)
                nn = tl.layers.Dense(n_units=300, act=tf.nn.relu6, W_init=w_init, name='la2')(nn)
                mu = tl.layers.Dense(n_units=N_A, act=tf.nn.tanh, W_init=w_init, name='mu')(nn)
                sigma = tl.layers.Dense(n_units=N_A, act=tf.nn.softplus, W_init=w_init, name='sigma')(nn)
            return tl.models.Model(inputs=ni, outputs=[mu, sigma], name=scope + '/Actor')

        self.actor = get_actor([None, N_S])
        self.actor.train()  # train mode for Dropout, BatchNorm

        def get_critic(input_shape):  # we use Value-function here, but not Q-function.
            with tf.name_scope(self.scope):
                ni = tl.layers.Input(input_shape, name='in')
                nn = tl.layers.Dense(n_units=500, act=tf.nn.relu6, W_init=w_init, name='lc')(ni)
                nn = tl.layers.Dense(n_units=300, act=tf.nn.relu6, W_init=w_init, name='lc2')(nn)
                v = tl.layers.Dense(n_units=1, W_init=w_init, name='v')(nn)
            return tl.models.Model(inputs=ni, outputs=v, name=scope + '/Critic')

        self.critic = get_critic([None, N_S])
        self.critic.train()  # train mode for Dropout, BatchNorm

    #@tf.function  # convert numpy functions to tf.Operations in the TFgraph, return tensor
    def update_global(self, buffer_s, buffer_a, buffer_v_target, globalAC):
        ''' update the global critic '''
        with tf.GradientTape() as tape_c:
            v = self.critic(buffer_s)  # shape: [batch_size, 1]
            td = buffer_v_target - v   # shape: [batch_size, 1]
            c_loss = tf.reduce_mean(tf.square(td))
        c_grads = tape_c.gradient(c_loss, self.critic.trainable_weights)
        OPT_C.apply_gradients(zip(c_grads, globalAC.critic.trainable_weights))

        ''' update the global actor '''
        with tf.GradientTape() as tape_a:
            mu, sigma = self.actor(buffer_s)  # shape: [batch_size, N_A]
            self.test = sigma[0]
            mu, sigma = mu * A_BOUND[1], sigma + 1e-5

            dist = tfd.Normal(mu, sigma)
            log_prob = dist.log_prob(buffer_a)  # shape: [batch_size, N_A]
            td_broadcast = tf.broadcast_to(td, tf.shape(log_prob))  # shape match
            exp_v = log_prob * td_broadcast                         # shape: [batch_size, N_A]
            exp_v = tf.reduce_sum(exp_v, axis=1)                    # shape: [batch_size]

            entropy = dist.entropy()                                # shape: [batch_size, N_A]
            entropy = tf.reduce_sum(entropy, axis=1)                # shape: [batch_size]

            a_loss = tf.reduce_mean(-ENTROPY_BETA * entropy + exp_v)

        a_grads = tape_a.gradient(a_loss, self.actor.trainable_weights)
        OPT_A.apply_gradients(zip(a_grads, globalAC.actor.trainable_weights))
        return self.test

    #@tf.function
    def pull_global(self, globalAC):  # run by a local, pull weights from the global nets
        for l_p, g_p in zip(self.actor.trainable_weights, globalAC.actor.trainable_weights):
            l_p.assign(g_p)
        for l_p, g_p in zip(self.critic.trainable_weights, globalAC.critic.trainable_weights):
            l_p.assign(g_p)

    def get_action(self, s, greedy=False):  # run by a local
        s = s[np.newaxis, :]
        self.mu, self.sigma = self.actor(s)

        with tf.name_scope('wrap_a_out'):
            self.mu, self.sigma = self.mu * A_BOUND[1], self.sigma + 1e-5

        if greedy:
            return self.mu.numpy()[0]

        normal_dist = tfd.Normal(self.mu, self.sigma)  # for continuous action space
        sample = normal_dist.sample()  # shape: [1, 4]
        sample = tf.squeeze(sample, axis=0)  # shape: [4]
        low = tf.convert_to_tensor(A_BOUND[0][0], dtype=tf.float32)  # shape: [4]
        high = tf.convert_to_tensor(A_BOUND[1][0], dtype=tf.float32) # shape: [4]
        self.A = tf.clip_by_value(sample, low, high)
        return self.A.numpy()


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

class Worker(object):
    def __init__(self, name):
        self.env = gym.make(ENV_ID, render_mode="rgb_array")
        self.name = name
        self.AC = ACNet(name)

    def work(self, globalAC):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while GLOBAL_EP < MAX_GLOBAL_EP:
            s, _ = self.env.reset()
            ep_r = 0
            while True:
                s = s.astype('float32')
                a = self.AC.get_action(s)
                a = np.array(a).reshape(-1)  # ← اصلاح نهایی برای جلوگیری از TypeError
                s_, r, terminated, truncated, _ = self.env.step(a)
                done = terminated or truncated
                s_ = s_.astype('float32')
                if r == -100: r = -2
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    v_s_ = 0 if done else self.AC.critic(s_[np.newaxis, :])[0, 0]
                    buffer_v_target = []
                    for r in reversed(buffer_r):
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()
                    bs = tf.convert_to_tensor(np.vstack(buffer_s), dtype=tf.float32)
                    ba = tf.convert_to_tensor(np.vstack(buffer_a), dtype=tf.float32)
                    bv = tf.convert_to_tensor(np.vstack(buffer_v_target), dtype=tf.float32)
                    self.AC.update_global(bs, ba, bv, globalAC)
                    self.AC.pull_global(globalAC)
                    buffer_s, buffer_a, buffer_r = [], [], []

                s = s_
                total_step += 1
                if done:
                    GLOBAL_RUNNING_R.append(ep_r if not GLOBAL_RUNNING_R else 0.95 * GLOBAL_RUNNING_R[-1] + 0.05 * ep_r)
                    print(f'Training | {self.name}, Episode: {GLOBAL_EP}/{MAX_GLOBAL_EP}, Reward: {ep_r:.2f}, Time: {time.time() - T0:.2f}')
                    GLOBAL_EP += 1
                    break


if __name__ == "__main__":

    env = gym.make(ENV_ID, render_mode="rgb_array")

    # reproducible
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    N_S = env.observation_space.shape[0]
    N_A = env.action_space.shape[0]

    A_BOUND = [env.action_space.low, env.action_space.high]
    A_BOUND[0] = A_BOUND[0].reshape(1, N_A)
    A_BOUND[1] = A_BOUND[1].reshape(1, N_A)

    with tf.device("/cpu:0"):
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params

    T0 = time.time()
    if args.train:
        # ============================= TRAINING ===============================
        with tf.device("/cpu:0"):
            OPT_A = tf.compat.v1.train.RMSPropOptimizer(LR_A, name='RMSPropA')
            OPT_C = tf.compat.v1.train.RMSPropOptimizer(LR_C, name='RMSPropC')
            workers = []
            # Create worker
            for i in range(N_WORKERS):
                i_name = 'Worker_%i' % i  # worker name
                workers.append(Worker(i_name))

        COORD = tf.train.Coordinator()

        # start TF threading
        worker_threads = []
        for worker in workers:
            job = lambda: worker.work(GLOBAL_AC)
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        COORD.join(worker_threads)

        GLOBAL_AC.save()

        plt.plot(GLOBAL_RUNNING_R)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

    if args.test:
        # ============================= EVALUATION =============================
        GLOBAL_AC.load()
        for episode in range(TEST_EPISODES):
            s, _ = env.reset()
            episode_reward = 0
            while True:
                env.render()
                s = s.astype('float32')  # double to float
                a = GLOBAL_AC.get_action(s, greedy=True)
                s_, r, terminated, truncated, _ = env.step(a)
                d = terminated or truncated
                episode_reward += r
                if d:
                    break
            print(
                'Testing  | Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                    episode + 1, TEST_EPISODES, episode_reward,
                    time.time() - T0
                )
            )
