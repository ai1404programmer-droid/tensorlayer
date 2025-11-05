"""
Actor-Critic 
-------------
It uses TD-error as the Advantage.

Actor Critic History
----------------------
A3C > DDPG > AC

Advantage
----------
AC converge faster than Policy Gradient.

Disadvantage (IMPORTANT)
------------------------
The Policy is oscillated (difficult to converge), DDPG can solve
this problem using advantage of DQN.

Reference
----------
paper: https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf
View more on MorvanZhou's tutorial page: https://morvanzhou.github.io/tutorials/

Environment
------------
CartPole-v0: https://gym.openai.com/envs/CartPole-v0

A pole is attached by an un-actuated joint to a cart, which moves along a
frictionless track. The system is controlled by applying a force of +1 or -1
to the cart. The pendulum starts upright, and the goal is to prevent it from
falling over.

A reward of +1 is provided for every timestep that the pole remains upright.
The episode ends when the pole is more than 15 degrees from vertical, or the
cart moves more than 2.4 units from the center.


Prerequisites
--------------
tensorflow >=2.0.0a0
tensorlayer >=2.0.0

To run
------
python tutorial_AC.py --train/test

"""
import argparse
import time
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import gymnasium as gym

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # فقط خطاهای مهم رو نمایش بده
tl.logging.set_verbosity(tl.logging.DEBUG)

#####################  hyper parameters  ####################

ENV_ID = 'CartPole-v1'
RANDOM_SEED = 2
RENDER = False

ALG_NAME = 'AC'
TRAIN_EPISODES = 20
TEST_EPISODES = 10
MAX_STEPS = 50
LAM = 0.9
LR_A = 0.001
LR_C = 0.01


###############################  Actor-Critic  ####################################

class Actor(object):
    def __init__(self, state_dim, action_num, lr=0.001):
        input_layer = tl.layers.Input([None, state_dim], name='state')
        layer = tl.layers.Dense(
            n_units=30, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden'
        )(input_layer)
        layer = tl.layers.Dense(n_units=action_num, name='actions')(layer)
        self.model = tl.models.Model(inputs=input_layer, outputs=layer, name="Actor")

        self.model.train()
        self.optimizer = tf.optimizers.Adam(lr)

    def learn(self, state, action, td_error):
        with tf.GradientTape() as tape:
            _logits = self.model(np.array([state]))
            _exp_v = tl.rein.cross_entropy_reward_loss(
                logits=_logits, actions=[action], rewards=td_error[0]
            )
        grad = tape.gradient(_exp_v, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
        return _exp_v

    def get_action(self, state, greedy=False):
        _logits = self.model(np.array([state]))
        _probs = tf.nn.softmax(_logits).numpy()
        if greedy:
            return np.argmax(_probs.ravel())
        return tl.rein.choice_action_by_probs(_probs.ravel())

    def save(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_npz_dict(self.model.trainable_weights, name=os.path.join(path, 'model_actor.npz'))

    def load(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_and_assign_npz_dict(name=os.path.join(path, 'model_actor.npz'), network=self.model)


class Critic(object):
    def __init__(self, state_dim, lr=0.01):
        input_layer = tl.layers.Input([None, state_dim], name='state')
        layer = tl.layers.Dense(
            n_units=30, act=tf.nn.relu6, W_init=tf.random_uniform_initializer(0, 0.01), name='hidden'
        )(input_layer)
        layer = tl.layers.Dense(n_units=1, act=None, name='value')(layer)
        self.model = tl.models.Model(inputs=input_layer, outputs=layer, name="Critic")
        self.model.train()

        self.optimizer = tf.optimizers.Adam(lr)

    def learn(self, state, reward, state_, done):
        d = 0 if done else 1
        v_ = self.model(np.array([state_]))
        with tf.GradientTape() as tape:
            v = self.model(np.array([state]))
            td_error = reward + d * LAM * v_ - v
            loss = tf.square(td_error)
        grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
        return td_error

    def save(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        if not os.path.exists(path):
            os.makedirs(path)
        tl.files.save_npz_dict(self.model.trainable_weights, name=os.path.join(path, 'model_critic.npz'))

    def load(self):
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_and_assign_npz_dict(name=os.path.join(path, 'model_critic.npz'), network=self.model)


###############################  Main ####################################

if __name__ == '__main__':

    args = argparse.Namespace()
    args.train = True   # برای آموزش
    args.test = False   # برای تست تغییر بده

    # محیط
    env = gym.make(ENV_ID, render_mode="human" if RENDER else None)

    # seed
    env.reset(seed=RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n

    print("observation dimension:", N_F)
    print("num of actions:", N_A)

    actor = Actor(state_dim=N_F, action_num=N_A, lr=LR_A)
    critic = Critic(state_dim=N_F, lr=LR_C)

    t0 = time.time()

    ####################### Train #######################
    if args.train:
        all_episode_reward = []
        for episode in range(TRAIN_EPISODES):
            state, info = env.reset()
            state = state.astype(np.float32)
            step = 0
            episode_reward = 0

            while True:
                if RENDER: env.render()

                action = actor.get_action(state)
                state_new, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state_new = state_new.astype(np.float32)

                if done:
                    reward = -20   # reward shaping

                episode_reward += reward

                td_error = critic.learn(state, reward, state_new, done)
                actor.learn(state, action, td_error)

                state = state_new
                step += 1

                if done or step >= MAX_STEPS:
                    break

            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1] * 0.9 + episode_reward * 0.1)

            print(f"Training | Episode: {episode+1}/{TRAIN_EPISODES} "
                  f"| Episode Reward: {episode_reward:.0f} "
                  f"| Running Time: {time.time()-t0:.2f}")

            if step >= MAX_STEPS:
                print("Early Stopping")
                break

        actor.save()
        critic.save()

        plt.plot(all_episode_reward)
        if not os.path.exists('image'):
            os.makedirs('image')
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

    ####################### Test #######################
    if args.test:
        actor.load()
        critic.load()
        for episode in range(TEST_EPISODES):
            state, info = env.reset()
            state = state.astype(np.float32)
            episode_reward = 0
            t = 0
            while True:
                env.render()
                action = actor.get_action(state, greedy=True)
                state_new, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state_new = state_new.astype(np.float32)

                episode_reward += reward
                state = state_new
                t += 1

                if done or t >= MAX_STEPS:
                    print(f"Testing | Episode: {episode+1}/{TEST_EPISODES} "
                          f"| Episode Reward: {episode_reward:.0f} "
                          f"| Running Time: {time.time()-t0:.2f}")
                    break
