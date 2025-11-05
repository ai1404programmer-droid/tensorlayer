#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
REINFORCE (Monte-Carlo Policy Network) for Pong-v5
with Gymnasium + AtariPreprocessing (grayscale)
"""
import os
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
import ale_py

tl.logging.set_verbosity(tl.logging.DEBUG)

# ================= hyperparameters =================
screen_size = 84
input_dim = screen_size * screen_size  # grayscale image -> 1 channel
H = 200                  # hidden layer size
batch_size = 10
learning_rate = 1e-4
gamma = 0.99
decay_rate = 0.99
render = True
model_file_name = "model_pong_v5.npz"

np.set_printoptions(threshold=np.inf)

# ================= Atari environment =================
gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="rgb_array")
env = AtariPreprocessing(
    env,
    noop_max=10,
    frame_skip=4,
    terminal_on_life_loss=True,
    screen_size=screen_size,
    grayscale_obs=True,     # یک کانال
    grayscale_newaxis=False # shape -> (84,84)
)

# ================= preprocess frame =================
def prepro(frame):
    """Flatten grayscale frame"""
    return frame.astype(np.float32).ravel()

# ================= policy network =================
def get_model(input_dim):
    ni = tl.layers.Input([None, input_dim], name='input')
    nn = tl.layers.Dense(H, act=tf.nn.relu, name='hidden')(ni)
    nn = tl.layers.Dense(3, name='output')(nn)  # 3 actions: stay, up, down
    return tl.models.Model(inputs=ni, outputs=nn, name='policy_net')

model = get_model(input_dim)
train_weights = model.trainable_weights
optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate, rho=decay_rate)
model.train()

# ================= save/load =================
def save_ckpt(model):
    os.makedirs('model', exist_ok=True)
    tl.files.save_npz_dict(model.trainable_weights, name=os.path.join('model', model_file_name))

def load_ckpt(model):
    path = os.path.join('model', model_file_name)
    if os.path.exists(path):
        tl.files.load_and_assign_npz_dict(name=path, network=model)
        print(f"[INFO] Loaded model from {path}")

# ================= training =================
observation, _ = env.reset()
prev_x = None
reward_sum = 0
episode_number = 0
game_number = 0
running_reward = None
MAX_EPISODES = 10
xs, ys, rs = [], [], []

start_time = time.time()

while episode_number < MAX_EPISODES:
    if render:
        env.render()

    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim, dtype=np.float32)
    x = x.reshape(1, input_dim)
    prev_x = cur_x

    logits = model(x)
    prob = tf.nn.softmax(logits).numpy()[0]

    # choose action: 1=stay, 2=up, 3=down
    action = tl.rein.choice_action_by_probs(prob, [1, 2, 3])
    observation, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    reward_sum += reward
    xs.append(x)
    ys.append(action - 1)  # adjust to 0-index
    rs.append(reward)

    if done:
        episode_number += 1
        game_number = 0

        if episode_number % batch_size == 0:
            print('Batch finished. Updating parameters...')
            epx = np.vstack(xs)
            epy = np.asarray(ys)
            epr = np.asarray(rs)

            discounted_r = tl.rein.discount_episode_rewards(epr, gamma)
            discounted_r -= np.mean(discounted_r)
            discounted_r /= np.std(discounted_r) + 1e-8

            xs, ys, rs = [], [], []

            with tf.GradientTape() as tape:
                logits = model(epx)
                loss = tl.rein.cross_entropy_reward_loss(logits, epy, discounted_r)
            grads = tape.gradient(loss, train_weights)
            optimizer.apply_gradients(zip(grads, train_weights))

            save_ckpt(model)

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print(f'Episode {episode_number} finished. Episode reward: {reward_sum}, Running mean: {running_reward:.2f}')
        reward_sum = 0
        observation, _ = env.reset()
        prev_x = None

    if reward != 0:
        print(f'Episode {episode_number}: game {game_number} reward {reward} time {time.time() - start_time:.2f}s',
              '' if reward == -1 else '!!!!!!!')
        start_time = time.time()
        game_number += 1
