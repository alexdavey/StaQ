from collections import namedtuple
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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


def aggregate(x, method):
    if method == 'mean':
        return x.mean(0)
    elif method == 'min':
        return x.min(0)[0]


def make_neural_model(netconf):
    # CNN for MinAtar observations
    insize = netconf['input_size']
    in_channels, H, W = insize[0], insize[1], insize[2]
    ops = []

    # Build CNN from config
    current_channels = in_channels
    for out_channels, ker_size, stride in zip(netconf['cnn_config']['channels'], netconf['cnn_config']['kernel_size'], netconf['cnn_config']['stride']):
        ops.extend([
            nn.Conv2d(current_channels, out_channels, kernel_size=ker_size, stride=stride),
            nn.ReLU()
        ])
        current_channels = out_channels

    # Flatten and get output size
    ops.append(nn.Flatten())
    cnn = nn.Sequential(*ops)

    with torch.no_grad():
        sample_input = torch.zeros(1, in_channels, H, W)
        n_flatten = cnn(sample_input).shape[1]

    # Add final linear layer
    ops.extend([nn.Linear(n_flatten, netconf['nb_neurones']), nn.ReLU(), zero_linear(nn.Linear(netconf['nb_neurones'], netconf['output_size']))])
    return nn.Sequential(*ops)


class StackedEnsemble(nn.Module):
    def __init__(self, anns, strides=None):
        super().__init__()
        # ann should be iterable List[torch.nn.Module]
        self.sweights = nn.ParameterList()  # stacked weights, list of len L (layers) with elements of dims: (Niter, in_dim, out_dim)
        self.sbiases = nn.ParameterList()
        self.non_lin = []
        self.layer_types = []  # Track the layer type ('conv' or 'mlp')
        self.strides = strides

        for idx, f in enumerate(anns[0]):
            if isinstance(f, nn.Linear) or isinstance(f, nn.Conv2d):
                weights, biases = [], []
                for ann in anns:
                    if isinstance(f, nn.Linear):
                        weights.append(ann[idx].weight.t().clone()[None, ...])
                        biases.append(ann[idx].bias.t().clone()[None, None, ...])  #! Do we need this, or will it broadcast just fine with only one None as in CNN case, or no None at all?
                    else:
                        weights.append(ann[idx].weight.clone()[None, ...])
                        biases.append(ann[idx].bias.clone()[None, ...])
                self.sweights.append(torch.cat(weights, dim=0))
                self.sbiases.append(torch.cat(biases, dim=0))
                if isinstance(f, nn.Linear):
                    self.layer_types.append('mlp')
                else:
                    self.layer_types.append('conv')
            elif isinstance(f, nn.Flatten):
                continue
            else:
                self.non_lin.append(f)
        if len(self.non_lin) < len(self.sweights):
            self.non_lin += [nn.Identity()] * (len(self.sweights) - len(self.non_lin))

    def forward(self, x):
        for layer_idx, (w, b, nl, layer_type) in enumerate(zip(self.sweights, self.sbiases, self.non_lin, self.layer_types)):
            if layer_type == 'conv':
                N, out_channels, in_channels, kernel_h, kernel_w = w.shape
                if layer_idx == 0:
                    x = F.conv2d(
                        x,
                        w.view(N * out_channels, in_channels, kernel_h, kernel_w),
                        b.view(-1),
                        stride=self.strides[layer_idx],
                        groups=1
                    )
                else:
                    x = F.conv2d(
                        x,
                        w.view(N * out_channels, in_channels, kernel_h, kernel_w),
                        b.view(-1),
                        stride=self.strides[layer_idx],
                        groups=N
                    )
                x = nl(x)

            elif layer_type == 'mlp':
                if len(x.shape) > 3:  # Flatten if needed
                    batch_size = x.size(0)
                    N = w.shape[0]
                    x = x.view(batch_size, N, -1).transpose(0, 1)

                matmul_result = torch.matmul(x, w)  # [Batch, Features] x [M, Features, Output_Size]
                x = nl(matmul_result + b)
        return x


class SamplerWithLogits(Sampler):
    # Sample transitions from gymnasium type environments
    def __init__(self, env, device='cpu', seed=None):
        super().__init__(env, device, seed)
        self.Trans = namedtuple('Trans', 'obs act rwd terminated truncated nobs nologits nact')  # order must match yield below

    def _rollout(self):
        obs, _ = self.env.reset(seed=self.seed)
        self.seed = None
        act, entr, _ = self.policy(obs)
        done = False
        step = 0
        while not done:
            step += 1
            nobs, rwd, terminated, truncated, _ = self.env.step(act)
            if np.abs(rwd) > self.max_abs_reward:
                self.max_abs_reward = np.abs(rwd)
            nact, nent, nlogits = self.policy(nobs)
            yield self.Trans(obs, act, rwd, terminated, truncated, nobs, nlogits, nact), entr, step
            obs = nobs
            act = nact
            entr = nent
            done = terminated or truncated


class ReplayMemoryWithLogits(ReplayMemory, torch.utils.data.Dataset):
    def __init__(self, max_size, state_shape, nb_act, device, obs_type=torch.float32):
        super().__init__(max_size, state_shape, device, obs_type)
        self.nb_act = nb_act
        self.ReplayMemorySamples = namedtuple('ReplayMemorySamples',
                                              ['obs', 'act', 'rwd', 'terminated', 'nobs', 'nologits'])
        # Handle both CNN and MLP input shapes
        if isinstance(state_shape, int):
            state_shape = (state_shape,)

        self.repmem = self.ReplayMemorySamples(
            obs=torch.zeros(max_size, *state_shape, device=self.device, dtype=obs_type),
            act=torch.zeros(max_size, 1, device=self.device).long(),
            rwd=torch.zeros(max_size, 1, device=self.device),
            terminated=torch.zeros(max_size, 1, device=self.device),
            nobs=torch.zeros(max_size, *state_shape, device=self.device, dtype=obs_type),
            nologits=torch.zeros(max_size, nb_act, device=self.device, dtype=torch.float32),
        )

    def sample_with_idxs(self, batch_size, device=None):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        if device is None:
            return self.ReplayMemorySamples(
                obs=self.repmem.obs[idxs],
                act=self.repmem.act[idxs],
                rwd=self.repmem.rwd[idxs],
                terminated=self.repmem.terminated[idxs],
                nobs=self.repmem.nobs[idxs],
                nologits=self.repmem.nologits[idxs],
            ), idxs
        else:
            return self.ReplayMemorySamples(
                obs=self.repmem.obs[idxs].to(device=device),
                act=self.repmem.act[idxs].to(device=device),
                rwd=self.repmem.rwd[idxs].to(device=device),
                terminated=self.repmem.terminated[idxs].to(device=device),
                nobs=self.repmem.nobs[idxs].to(device=device),
                nologits=self.repmem.nologits[idxs].to(device=device),
            ), idxs

    def _get_idexes(self, start_idx, end_idx):  # start and end indexes are relative to how old is the data
            assert start_idx < end_idx
            datastarts = self.write_idx if self.size == self.max_size else 0
            lidx = (datastarts + start_idx) % self.max_size
            ridx = (datastarts + end_idx) % self.max_size
            return lidx, ridx

    def get(self, key, start_idx=None, end_idx=None):  # start and end indexes are relative to how old is the data
            if start_idx is None and end_idx is None:
                return getattr(self.repmem, key)[:self.size]
            else:
                lidx, ridx = self._get_idexes(start_idx, end_idx)
                if lidx < ridx:
                    return getattr(self.repmem, key)[lidx:ridx]
                else:
                    return torch.cat([getattr(self.repmem, key)[lidx:],
                                      getattr(self.repmem, key)[:ridx]], dim=0)

    def set(self, key, tensor, start_idx=None, end_idx=None):  # start and end indexes are relative to how old is the data
        if start_idx is None and end_idx is None:
            getattr(self.repmem, key)[:self.size] = tensor
        else:
            lidx, ridx = self._get_idexes(start_idx, end_idx)
            if lidx < ridx:
                getattr(self.repmem, key)[lidx:ridx] = tensor
            else:
                a = getattr(self.repmem, key)
                a[lidx:] = tensor[:self.size - lidx]
                a[:ridx] = tensor[self.size - lidx:]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return (self.repmem.obs[idx], self.repmem.act[idx], self.repmem.rwd[idx],
                self.repmem.terminated[idx], self.repmem.nobs[idx], self.repmem.nologits[idx])


class ChannelFirstWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(old_shape[2], old_shape[0], old_shape[1]),  # HWC -> CHW
            dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._convert_obs(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._convert_obs(obs), reward, term, trunc, info

    def _convert_obs(self, obs):
        return np.transpose(obs, (2, 0, 1))


def make_env(
        env_name,
        reward_scale=1.0,
        env_constructor=gym.make,
        env_render_mode=None,
        env_eval_render_mode=None,
        use_cnn=False,
):
    env = gym.make(env_name, render_mode=env_render_mode)
    env_eval = env_constructor(env_name, render_mode=env_eval_render_mode)

    if env_name.startswith('MinAtar'):
        if use_cnn:
            env = ChannelFirstWrapper(env)
            env_eval = ChannelFirstWrapper(env_eval)
        else:
            env = FlattenObservation(env)
            env_eval = FlattenObservation(env_eval)

    return env, env_eval


def get_obs_shape(env_name):
    env = gym.make(env_name)
    obs_shape = env.observation_space.shape
    return obs_shape[0], obs_shape[1], obs_shape[2]


class MinatarCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
