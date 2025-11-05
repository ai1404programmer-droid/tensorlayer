"""
Vanilla Policy Gradient (VPG / REINFORCE) with Gymnasium compatibility
"""
import argparse
import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl

parser = argparse.ArgumentParser(description='Train or test PG agent.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=True)
args = parser.parse_args()

##################### Hyperparameters ####################

ENV_ID = 'CartPole-v1'
RANDOM_SEED = 1
RENDER = False

ALG_NAME = 'PG'
TRAIN_EPISODES = 10
TEST_EPISODES = 10
MAX_STEPS = 10

##################### PG Class ####################

class PolicyGradient:
    def __init__(self, state_dim, action_num, learning_rate=0.02, gamma=0.99):
        self.gamma = gamma
        self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []

        input_layer = tl.layers.Input([None, state_dim], tf.float32)
        layer = tl.layers.Dense(
            n_units=30, act=tf.nn.tanh,
            W_init=tf.random_normal_initializer(mean=0, stddev=0.3),
            b_init=tf.constant_initializer(0.1)
        )(input_layer)
        all_act = tl.layers.Dense(
            n_units=action_num, act=None,
            W_init=tf.random_normal_initializer(mean=0, stddev=0.3),
            b_init=tf.constant_initializer(0.1)
        )(layer)

        self.model = tl.models.Model(inputs=input_layer, outputs=all_act)
        self.model.train()
        self.optimizer = tf.optimizers.Adam(learning_rate)

    def get_action(self, s, greedy=False):
        logits = self.model(np.array([s], np.float32))
        probs = tf.nn.softmax(logits).numpy().ravel()
        if greedy:
            return np.argmax(probs)
        return tl.rein.choice_action_by_probs(probs)

    def store_transition(self, s, a, r):
        self.state_buffer.append(np.array([s], np.float32))
        self.action_buffer.append(a)
        self.reward_buffer.append(r)

    def learn(self):
        discounted_rewards = self._discount_and_norm_rewards()
        with tf.GradientTape() as tape:
            logits = self.model(np.vstack(self.state_buffer))
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=np.array(self.action_buffer)
            )
            loss = tf.reduce_mean(neg_log_prob * discounted_rewards)
        grad = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_weights))
        self.state_buffer, self.action_buffer, self.reward_buffer = [], [], []
        return discounted_rewards

    def _discount_and_norm_rewards(self):
        discounted = np.zeros_like(self.reward_buffer, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(self.reward_buffer))):
            running_add = running_add * self.gamma + self.reward_buffer[t]
            discounted[t] = running_add
        discounted -= np.mean(discounted)
        discounted /= (np.std(discounted) + 1e-8)
        return discounted
        
    def save(self):
        """
        save trained weights (npz format)
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        os.makedirs(path, exist_ok=True)
        tl.files.save_npz_dict(self.model.trainable_weights, name=os.path.join(path, 'pg_policy.npz'))

    def load(self):
        """
        load trained weights (npz format)
        """
        path = os.path.join('model', '_'.join([ALG_NAME, ENV_ID]))
        tl.files.load_and_assign_npz_dict(name=os.path.join(path, 'pg_policy.npz'), network=self.model)

##################### Main ####################

if __name__ == '__main__':
    env = gym.make(ENV_ID)

    # reproducible
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    env.reset(seed=RANDOM_SEED)

    agent = PolicyGradient(
        action_num=env.action_space.n,
        state_dim=env.observation_space.shape[0]
    )

    t0 = time.time()

    if args.train:
        all_episode_reward = []
        for episode in range(TRAIN_EPISODES):
            state, _ = env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                if RENDER:
                    env.render()
                action = agent.get_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                agent.store_transition(state, action, reward)
                state = next_state
                episode_reward += reward
                if done:
                    break
            agent.learn()
            print(f'Training  | Episode: {episode+1}/{TRAIN_EPISODES}  | Reward: {episode_reward}  | Time: {time.time()-t0:.2f}')
            if episode == 0:
                all_episode_reward.append(episode_reward)
            else:
                all_episode_reward.append(all_episode_reward[-1]*0.9 + episode_reward*0.1)
        agent.save()
        plt.plot(all_episode_reward)
        os.makedirs('image', exist_ok=True)
        plt.savefig(os.path.join('image', '_'.join([ALG_NAME, ENV_ID])))

    if args.test:
        agent.load()
        for episode in range(TEST_EPISODES):
            state, _ = env.reset()
            episode_reward = 0
            for step in range(MAX_STEPS):
                if RENDER:
                    env.render()
                action = agent.get_action(state, greedy=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state
                episode_reward += reward
                if done:
                    break
            print(f'Testing  | Episode: {episode+1}/{TEST_EPISODES}  | Reward: {episode_reward}  | Time: {time.time()-t0:.2f}')
