"""
Env wrappers compatible with Gymnasium 1.x, TF 2.x, TensorLayer 2.x

Adapted from:
- https://pypi.org/project/gym-vec-env
- https://github.com/openai/baselines/blob/master/baselines/common/*wrappers.py
"""
from collections import deque
from functools import partial
from multiprocessing import Pipe, Process, cpu_count
from sys import platform
import ale_py

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

__all__ = (
    'build_env',
    'TimeLimit',
    'NoopResetEnv',
    'FireResetEnv',
    'EpisodicLifeEnv',
    'MaxAndSkipEnv',
    'ClipRewardEnv',
    'WarpFrame',
    'FrameStack',
    'LazyFrames',
    'RewardScaler',
    'SubprocVecEnv',
    'VecFrameStack',
    'Monitor',
    'NormalizedActions',
)

cv2.ocl.setUseOpenCL(False)

# env_id -> env_type
id2type = {}
for env_spec in gym.envs.registry.values():
    entry_point = getattr(env_spec, "entry_point", None)
    if isinstance(entry_point, str) and "atari" in entry_point.lower():
        id2type[env_spec.id] = "atari"
    else:
        # همه‌ی دیگر محیط‌ها را classic_control فرض می‌کنیم
        id2type[env_spec.id] = "classic_control"


def build_env(env_id, vectorized=False, seed=0, reward_scale=1.0, nenv=0):
    env_type = id2type[env_id]
    nenv = nenv or cpu_count() // (1 + (platform == 'darwin'))
    stack = env_type == 'atari'
    if not vectorized:
        env = _make_env(env_id, env_type, seed, reward_scale, stack)
    else:
        env = _make_vec_env(env_id, env_type, nenv, seed, reward_scale, stack)
    return env

from gymnasium.wrappers import AtariPreprocessing

def _make_env(env_id, env_type, seed, reward_scale, frame_stack=True):
    if env_type == 'atari':
        # نسخه جدید محیط Atari
        env = gym.make(env_id, render_mode=None, frameskip=1)
        env = AtariPreprocessing(
            env,
            noop_max=30,
            frame_skip=4,
            terminal_on_life_loss=True,
            screen_size=84,
            grayscale_obs=True,
            grayscale_newaxis=True  # خروجی shape -> (84,84,1)
        )
        env = Monitor(env)
        if reward_scale != 1:
            env = RewardScaler(env, reward_scale)
        if frame_stack:
            env = FrameStack(env, 4)
    elif env_type == 'classic_control':
        env = Monitor(gym.make(env_id))
        if reward_scale != 1:
            env = RewardScaler(env, reward_scale)
    else:
        raise NotImplementedError(f"Env type {env_type} not implemented")
    
    env.reset(seed=seed)
    return env

def _make_vec_env(env_id, env_type, nenv, seed, reward_scale, frame_stack=True):
    env = SubprocVecEnv(
        [partial(_make_env, env_id, env_type, seed + i, reward_scale, False) for i in range(nenv)]
    )
    if frame_stack and env_type == 'atari':
        env = VecFrameStack(env, 4)
    return env


# ====================== Wrappers ======================

class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        done = terminated or truncated
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = not terminated
        return obs, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        obs, info = self.env.reset(**kwargs)
        return obs, info


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = self.override_num_noops or self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            obs, reward, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, reward, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, reward, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, reward, terminated, truncated, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        shape = (self.height, self.width, 1 if self.grayscale else 3)
        self.observation_space = spaces.Box(0, 255, shape, dtype=np.uint8)

    def observation(self, frame):
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        shape = shp[:-1] + (shp[-1] * k,)
        self.observation_space = spaces.Box(0, 255, shape, dtype=env.observation_space.dtype)

    def reset(self):
        ob, info = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return np.array(self._get_ob()), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return np.array(self._get_ob()), reward, terminated, truncated, info

    def _get_ob(self):
        return LazyFrames(list(self.frames))


class LazyFrames:
    def __init__(self, frames):
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class RewardScaler(gym.RewardWrapper):
    def __init__(self, env, scale=0.01):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


# ====================== Vectorized Env ======================

def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            if done:
                obs, info = env.reset()
            remote.send((obs, reward, done, info))
        elif cmd == 'reset':
            obs, info = env.reset()
            remote.send(obs)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv:
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        self.observation_space, self.action_space = self.remotes[0].recv()

    def _step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def _step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step(self, actions):
        self._step_async(actions)
        return self._step_wait()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class VecFrameStack:
    def __init__(self, env, k):
        self.env = env
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        shape = shp[:-1] + (shp[-1] * k,)
        self.observation_space = spaces.Box(0, 255, shape, dtype=env.observation_space.dtype)
        self.action_space = env.action_space

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return np.array(self._get_ob())

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return np.array(self._get_ob()), reward, done, info

    def _get_ob(self):
        return LazyFrames(list(self.frames))


# ====================== Monitor ======================

class Monitor(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._monitor_rewards = []

    def reset(self, **kwargs):
        self._monitor_rewards = []
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._monitor_rewards.append(reward)
        done = terminated or truncated
        if done:
            info['episode'] = {'r': sum(self._monitor_rewards), 'l': len(self._monitor_rewards)}
        return obs, reward, terminated, truncated, info


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high
        scaled = low + (action + 1.0) * 0.5 * (high - low)
        return np.clip(scaled, low, high)

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high
        unscaled = 2 * (action - low) / (high - low) - 1
        return np.clip(unscaled, low, high)

def unit_test():
    # ---------------- Classic Control ----------------
    env_id = 'CartPole-v1'
    unwrapped_env = gym.make(env_id)
    wrapped_env = build_env(env_id, vectorized=False)

    obs, info = wrapped_env.reset()
    print('Reset {} observation shape {}'.format(env_id, obs.shape))

    done = False
    while not done:
        action = unwrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        done = terminated or truncated
        print(f'Take action {action} get reward {reward} done {done} info {info}')

    # ---------------- Atari Vectorized ----------------
    env_id = "ALE/Pong-v5"  # نام جدید Atari
    nenv = 2
    unwrapped_env = gym.make(env_id, render_mode="rgb_array", frameskip=1)
    
    wrapped_env = build_env(env_id, vectorized=True, nenv=nenv)

    obs, info = wrapped_env.reset()
    print('Reset {} observation shape {}'.format(env_id, obs.shape))

    for _ in range(10):
        actions = [unwrapped_env.action_space.sample() for _ in range(nenv)]
        actions = np.array(actions, dtype='int64')
        obs, rewards, dones, infos = wrapped_env.step(actions)
        print(f'Take actions {actions} get rewards {rewards} dones {dones}')

if __name__ == '__main__':
    unit_test()