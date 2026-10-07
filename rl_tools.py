from collections import namedtuple

import gymnasium as gym
import numpy as np
import torch


def zero_linear(f):
    f.weight.data.zero_()
    if f.bias is not None:
        f.bias.data.zero_()
    return f

def stable_kl_div(old_probs, new_probs, epsilon=1e-12, lib=torch):
    kl = new_probs * (lib.log(new_probs + epsilon) - lib.log(old_probs + epsilon))
    return kl


class ChannelsFirst(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (shape[2], shape[0], shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return observation.swapaxes(0, 2)


class Sampler:
    # Sample transitions from gymnasium type environments
    def __init__(self, env, device='cpu', seed=None):
        self.curr_rollout = []
        self.policy = None
        self.env = env
        self.Return = namedtuple('Return', 'global_step value step')
        self.max_abs_reward = 0
        self.curr_return = 0
        self.curr_entropy = 0.
        self.total_step = 0
        self.Trans = namedtuple('Trans', 'obs act rwd terminated truncated nobs nact')  # order must match yield below
        self.device = device
        self.seed = seed  # seeds the environment on the first reset only

    def _rollout(self):
        # Generates SARSA type transitions until episode's end
        obs, _ = self.env.reset(seed=self.seed)
        self.seed = None
        act, entr = self.policy(obs)
        done = False
        step = 0
        while not done:
            step += 1
            nobs, rwd, terminated, truncated, _ = self.env.step(act)
            if np.abs(rwd) > self.max_abs_reward:
                self.max_abs_reward = np.abs(rwd)
            nact, nent = self.policy(nobs)
            yield self.Trans(obs, act, rwd, terminated, truncated, nobs, nact), entr, step
            obs = nobs
            act = nact
            entr = nent
            done = terminated or truncated

    def rollouts(self, policy, min_trans, max_trans, returns_only=False):
        # Keep generating full trajectories until min_trans transitions are collected.
        # Specifying max_trans < inf can stop data collection before trajectory's end.
        # If min_trans = max_trans, will collect exactly min_trans transitions.
        assert (min_trans <= max_trans)
        returns = []
        entropies = []
        self.policy = policy
        all_trans = []
        # Generating transitions and computing returns
        gathered_steps = 0
        while gathered_steps < min_trans:
            for trans, entr, step in self.curr_rollout:
                if not returns_only:
                    all_trans.append(trans)
                self.curr_return += trans.rwd
                self.curr_entropy += entr
                self.total_step += 1
                gathered_steps += 1
                if trans.truncated or trans.terminated:
                    returns.append(self.Return(self.total_step, self.curr_return, step))
                    entropies.append(self.curr_entropy)
                    self.curr_return = 0
                    self.curr_entropy = 0
                if gathered_steps >= max_trans:
                    break
            if not gathered_steps >= max_trans:
                self.curr_rollout = self._rollout()

        # Saving into a dictionary of 2D torch.FloatTensor
        if returns_only:
            return None, returns, entropies
        else:
            paths = {}
            for key in set(self.Trans._fields):
                paths[key] = torch.tensor(np.asarray([getattr(t, key) for t in all_trans]), device=self.device, dtype=torch.float)
                if paths[key].ndim == 1:
                    paths[key] = paths[key][:, None]
            return paths, returns, entropies


class ReplayMemory:
    def __init__(self, max_size, state_shape, device, obs_type=torch.float32):
        self.device = device
        self.max_size = max_size
        self.size = 0
        self.write_idx = 0
        self.ReplayMemorySamples = namedtuple('ReplayMemorySamples',
                                              ['obs', 'act', 'rwd', 'terminated', 'nobs'])

        # Handle both CNN and MLP input shapes
        if isinstance(state_shape, int):
            state_shape = (state_shape,)

        self.repmem = self.ReplayMemorySamples(
            obs=torch.zeros(max_size, *state_shape, device=self.device, dtype=obs_type),
            act=torch.zeros(max_size, 1, device=self.device).long(),
            rwd=torch.zeros(max_size, 1, device=self.device),
            terminated=torch.zeros(max_size, 1, device=self.device),
            nobs=torch.zeros(max_size, *state_shape, device=self.device, dtype=obs_type),
        )

    def add_trans(self, trans):
        add_len = len(trans['rwd'])
        overflow = add_len + self.write_idx > self.max_size
        len_first_copy = self.max_size - self.write_idx
        data = self.repmem._asdict()
        for k in data:
            if k == 'act':
                v = trans[k].long()
            else:
                v = trans[k]
            if overflow:
                data[k][self.write_idx:] = v[:len_first_copy]
                data[k][0:add_len - len_first_copy] = v[len_first_copy:]
            else:
                data[k][self.write_idx:self.write_idx + add_len] = v
        self.write_idx = (self.write_idx + add_len) % self.max_size
        self.size = min(self.size + add_len, self.max_size)

    def sample_with_idxs(self, batch_size, device=None):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        if device is None:
            return self.ReplayMemorySamples(
                obs=self.repmem.obs[idxs].to(dtype=torch.float32),
                act=self.repmem.act[idxs],
                rwd=self.repmem.rwd[idxs],
                terminated=self.repmem.terminated[idxs],
                nobs=self.repmem.nobs[idxs].to(dtype=torch.float32),
            ), idxs
        else:
            return self.ReplayMemorySamples(
                obs=self.repmem.obs[idxs].to(device=device, dtype=torch.float32),
                act=self.repmem.act[idxs].to(device=device),
                rwd=self.repmem.rwd[idxs].to(device=device),
                terminated=self.repmem.terminated[idxs].to(device=device),
                nobs=self.repmem.nobs[idxs].to(device=device, dtype=torch.float32),
            ), idxs

    def sample(self, batch_size, device=None):
        return self.sample_with_idxs(batch_size, device)[0]
