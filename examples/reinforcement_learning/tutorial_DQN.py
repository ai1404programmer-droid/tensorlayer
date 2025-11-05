"""
Deep Q-Network Q(a, s) - Gymnasium compatible
---------------------------------------------
TD Learning, Off-Policy, e-Greedy Exploration (GLIE).
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
parser = argparse.ArgumentParser(description='Train or test DQN agent.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

tl.logging.set_verbosity(tl.logging.DEBUG)

# ------------------- Hyperparameters -------------------
env_id = 'FrozenLake-v1'  # Gymnasium updated environment
alg_name = 'DQN'
lambd = 0.99  # discount factor
e = 0.1       # e-Greedy Exploration
num_episodes = 10
render = False

# ------------------- Utilities -------------------
def to_one_hot(i, n_classes=16):
    a = np.zeros(n_classes, dtype=np.float32)
    a[i] = 1.0
    return a

# ------------------- Q-Network -------------------
def get_model(input_shape):
    ni = tl.layers.Input(input_shape, name='observation')
    nn = tl.layers.Dense(4, act=None,
                         W_init=tf.random_uniform_initializer(0, 0.01),
                         b_init=None,
                         name='q_a_s')(ni)
    return tl.models.Model(inputs=ni, outputs=nn, name="Q-Network")

# ------------------- Save/Load اصلاح شده -------------------
def save_ckpt(model):
    path = os.path.join('model', f'{alg_name}_{env_id}')
    os.makedirs(path, exist_ok=True)
    tl.files.save_npz_dict(model.trainable_weights, name=os.path.join(path, 'dqn_weights.npz'))

def load_ckpt(model):
    path = os.path.join('model', f'{alg_name}_{env_id}')
    tl.files.load_and_assign_npz_dict(name=os.path.join(path, 'dqn_weights.npz'), network=model)

# ------------------- Main -------------------
if __name__ == '__main__':
    qnetwork = get_model([None, 16])
    qnetwork.train()
    train_weights = qnetwork.trainable_weights

    optimizer = tf.optimizers.SGD(learning_rate=0.1)
    env = gym.make(env_id)

    t0 = time.time()
    all_episode_reward = []

    if args.train:
        for i in range(num_episodes):
            s, _ = env.reset()  # Gymnasium: reset() -> obs, info
            rAll = 0
            if render: env.render()

            for j in range(99):
                # Q values
                allQ = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
                a = np.argmax(allQ, 1)

                # e-Greedy
                if np.random.rand() < e:
                    a[0] = env.action_space.sample()

                # Step in environment
                s1, r, terminated, truncated, info = env.step(a[0])
                d = terminated or truncated
                if render: env.render()

                # Target Q
                Q1 = qnetwork(np.asarray([to_one_hot(s1, 16)], dtype=np.float32)).numpy()
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0, a[0]] = r + lambd * maxQ1

                # Train
                with tf.GradientTape() as tape:
                    _qvalues = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32))
                    _loss = tl.cost.mean_squared_error(targetQ, _qvalues, is_mean=False)
                grad = tape.gradient(_loss, train_weights)
                optimizer.apply_gradients(zip(grad, train_weights))

                rAll += r
                s = s1

                if d:
                    e = 1. / ((i / 50) + 10)  # GLIE: reduce epsilon
                    break

            # Reward tracking
            if i == 0:
                all_episode_reward.append(rAll)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + rAll * 0.1)

            print(f'Training  | Episode: {i+1}/{num_episodes}  | Reward: {rAll:.4f}  | Time: {time.time()-t0:.2f}')

        save_ckpt(qnetwork)
        plt.plot(all_episode_reward)
        os.makedirs('image', exist_ok=True)
        plt.savefig(os.path.join('image', f'{alg_name}_{env_id}.png'))

    if args.test:
        load_ckpt(qnetwork)
        for i in range(num_episodes):
            s, _ = env.reset()
            rAll = 0
            if render: env.render()

            for j in range(99):
                allQ = qnetwork(np.asarray([to_one_hot(s, 16)], dtype=np.float32)).numpy()
                a = np.argmax(allQ, 1)  # greedy
                s1, r, terminated, truncated, info = env.step(a[0])
                d = terminated or truncated
                rAll += r
                s = s1
                if render: env.render()
                if d: break

            print(f'Testing  | Episode: {i+1}/{num_episodes}  | Reward: {rAll:.4f}  | Time: {time.time()-t0:.2f}')
