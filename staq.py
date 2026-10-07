import argparse
from copy import deepcopy
from os.path import join
from time import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.wrappers import TimeLimit
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from rl_tools import ChannelsFirst, ReplayMemory, Sampler, stable_kl_div, zero_linear


class StackedNN:
    """Stacked NN layers for an efficient batched forward pass."""

    def __init__(self, ann, max_size, strides=None):
        # ann should be iterable List[torch.nn.Module]
        self.sweights = []  # stacked weights, list of len L (layers) with elements of dims: (Niter, in_dim, out_dim)
        self.sbiases = []
        self.non_lin = []
        self.layer_types = []  # Track the layer type ('conv' or 'mlp')
        self.ensemble_size = 1
        self.strides = strides
        self.max_size = max_size

        for f in ann:
            if isinstance(f, nn.Linear):
                self.sweights.append(f.weight.t().detach().clone()[None, ...])
                self.sbiases.append(f.bias.detach().clone()[None, None, ...])
                self.layer_types.append('mlp')
            elif isinstance(f, nn.Conv2d):
                self.sweights.append(f.weight.detach().clone()[None, ...])
                self.sbiases.append(f.bias.detach().clone()[None, ...])
                self.layer_types.append('conv')
            elif isinstance(f, nn.Flatten):
                continue
            else:
                self.non_lin.append(f)
        if len(self.non_lin) < len(self.sweights):
            self.non_lin += [nn.Identity()] * (len(self.sweights) - len(self.non_lin))

    def __call__(self, x):
        for layer_idx, (w, b, nl, layer_type) in enumerate(zip(self.sweights, self.sbiases, self.non_lin, self.layer_types)):

            if layer_type == 'conv':
                N, out_channels, in_channels, kernel_h, kernel_w = w.shape
                if N != self.ensemble_size:
                    self.ensemble_size = N

                # Broadcast on first layer
                if layer_idx == 0:
                    x = x.repeat(1, N, 1, 1)

                x = F.conv2d(
                    x,
                    w.view(N * out_channels, in_channels, kernel_h, kernel_w),
                    b.view(-1),
                    stride=self.strides[layer_idx],
                    groups=N
                )
                x = nl(x)

            elif layer_type == 'mlp':
                batch_size = x.size(0)
                if len(x.shape) > 3:  # Flatten if needed
                    N = self.ensemble_size
                    x = x.view(batch_size, N, -1).transpose(0, 1)

                matmul_result = torch.matmul(x, w)  # [Batch, M, Features] x [M, Features, Output_Size]
                x = nl(matmul_result + b)
        return x

    def push(self, ann):
        idx = 0
        for f in ann:
            if isinstance(f, nn.Linear):
                w = self.sweights[idx]
                b = self.sbiases[idx]
                if w.shape[0] > self.max_size:
                    w = w[1:]
                    b = b[1:]
                w_new = f.weight.t().detach().clone()[None, ...]
                b_new = f.bias.detach().clone()[None, None, ...]
                self.sweights[idx] = torch.cat((w, w_new), 0) if w.shape[0] > 0 else w_new
                self.sbiases[idx] = torch.cat((b, b_new), 0) if b.shape[0] > 0 else b_new
                idx += 1
            elif isinstance(f, nn.Conv2d):
                w = self.sweights[idx]
                b = self.sbiases[idx]
                if w.shape[0] > self.max_size:
                    w = w[1:]
                    b = b[1:]
                w_new = f.weight.detach().clone()[None, ...]
                b_new = f.bias.detach().clone()[None, ...]
                self.sweights[idx] = torch.cat((w, w_new), 0) if w.shape[0] > 0 else w_new
                self.sbiases[idx] = torch.cat((b, b_new), 0) if b.shape[0] > 0 else b_new
                idx += 1

    def decay(self, c):
        for w, b in zip(self.sweights, self.sbiases):
            w.data *= c
            b.data *= c

class StaQNet:
    def __init__(self, input_size, output_size, nb_hidden, hidden_width, memory_size, kl_weight, entropy_weight, use_w_correction,
                 device=torch.device('cpu'), nl=nn.ReLU(inplace=True), network_type="mlp", cnn_config=None):
        super().__init__()

        assert nb_hidden > 0, "Number of hidden layers must be greater than 0."
        self.nb_hidden = nb_hidden  # total number of hidden layers (remains fixed)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_width = hidden_width
        self.memory_size = memory_size
        self._kl_weight = kl_weight
        self._entropy_weight = entropy_weight
        self.eta = 1. / (kl_weight + entropy_weight)
        self.decay = kl_weight / (kl_weight + entropy_weight)
        self.use_w_correction = use_w_correction
        self.strides = None
        if self.use_w_correction:
            self.w_correction = 1. / (1. - self.decay ** self.memory_size)
        else:
            self.w_correction = 1.

        print(f'step size eta {self.eta} and decay {self.decay}')
        self.device = device
        self.nl = nl

        self.network_type = network_type
        self.network_config = cnn_config

        # Frozen net
        self.froz_feat = None
        # self.froz_q = None
        self.sig_q = None

        # Trainable mlp
        self.train_feat, self.train_q = self._get_new_mlps()

    def __call__(self, x):
        return self.train_q(self.train_feat(x))  # returns new Q

    def get_logits(self, x, no_old=False):  # get logits, that depend only on old features
        # no_old for debug only
        if self.froz_feat is None:  # no old features yet
            return torch.zeros(len(x), self.output_size, device=self.device)
        else:
            if no_old and self.sig_q.sweights[0].shape[0] == self.memory_size:  # debug only, for computing KL(pik, tilde{pik}) in the paper
                return self.sig_q(self.froz_feat(x))[1:].sum(0) * self.eta * self.w_correction
            return self.sig_q(self.froz_feat(x)).sum(0) * self.eta * self.w_correction  # it was easier to put eta here

    def _get_new_mlps(self):
        if self.network_type == "mlp":
            insize = self.input_size
            ops = []
            for _ in range(self.nb_hidden):
                ops.append(nn.Linear(insize, self.hidden_width))
                ops.append(self.nl)
                insize = self.hidden_width
            output_layer = zero_linear(nn.Linear(self.hidden_width, self.output_size).to(self.device))

            return nn.Sequential(*ops).to(self.device), output_layer
        else:
            insize = self.input_size
            in_channels = insize[0]
            H, W = insize[1], insize[2]
            ops = []

            # Build CNN from config
            current_channels = in_channels
            self.strides = self.network_config.get('stride', [2, 2])
            for out_channels, ker_size, stride in zip(self.network_config.get('channels', [32, 64]), self.network_config.get('kernel_size', [3, 3]),
                                                      self.network_config.get('stride', [2, 2])):
                ops.extend([
                    nn.Conv2d(current_channels, out_channels,
                              kernel_size=ker_size, stride=stride),
                    self.nl
                ])
                current_channels = out_channels

            # Flatten and get output size
            ops.append(nn.Flatten())
            cnn = nn.Sequential(*ops).to(self.device)

            with torch.no_grad():
                sample_input = torch.zeros(1, in_channels, H, W).to(self.device)
                n_flatten = cnn(sample_input).shape[1]

            # Add final linear layer
            ops.append(nn.Linear(n_flatten, self.hidden_width))
            ops.append(self.nl)

            output_layer = zero_linear(nn.Linear(self.hidden_width, self.output_size).to(self.device))

            return nn.Sequential(*ops).to(self.device), output_layer

    def parameters(self):
        if self.train_feat is not None:
            return [*self.train_feat.parameters(), *self.train_q.parameters()]

    def train(self, train_mode):
        if self.train_feat is not None:
            self.train_feat.train(train_mode)

    def set_entropy_weight(self, weight):
        self._entropy_weight = weight
        self.eta = 1. / (self._kl_weight + self._entropy_weight)
        self.decay = self._kl_weight / (self._kl_weight + self._entropy_weight)
        if self.use_w_correction:
            self.w_correction = 1. / (1. - self.decay ** self.memory_size)
        else:
            self.w_correction = 1.

    def update_sigq(self, decay=True):
        # merge frozen and trainable MLPs and delete oldest function if necessary
        self.train_feat.train(False)
        if self.froz_feat is None:  # building first frozen feat network
            self.froz_feat = StackedNN(self.train_feat, self.memory_size, strides=self.strides)
            self.sig_q = StackedNN([self.train_q], self.memory_size)
        else:
            self.froz_feat.push(self.train_feat)
            if decay:
                self.sig_q.decay(self.decay)
            self.sig_q.push([self.train_q])

def logits(dwexnn, obs, device):
    return dwexnn.get_logits(torch.tensor(obs[None, :].astype(np.float32), device=device))


def get_logits_ensemble(dwexnns, obs, device):
    return sum([logits(dwexnn, obs, device) for dwexnn in dwexnns]) / len(dwexnns)


def get_logits_ensemble_torch(dwexnns, obs, no_old=False):
    return sum([dwexnn.get_logits(obs, no_old) for dwexnn in dwexnns]) / len(dwexnns)


def numpy_softmax(obs, dwexnns, device):
    with torch.no_grad():
        dist = torch.distributions.Categorical(logits=get_logits_ensemble(dwexnns, obs, device))
        return dist.sample().squeeze().cpu().numpy(), dist.entropy().squeeze().cpu().numpy()


def numpy_argmax_policy(obs, dwexnns, device, ensemble=True):
    if ensemble:
        with torch.no_grad():
            dist = torch.distributions.Categorical(logits=get_logits_ensemble(dwexnns, obs, device))
    return dist.probs.argmax(1).squeeze().cpu().numpy(), dist.entropy().squeeze().cpu().numpy()


def numpy_egreedy_softpolicy(obs, dwexnns, eps, device):
    with torch.no_grad():
        act, entrop = numpy_softmax(obs, dwexnns, device)
    if np.random.rand() < eps:
        return np.random.randint(dwexnns[0].output_size), entrop
    return act, entrop


def update_target(sources, targets=None, tau=None, update_type=None):
    if update_type == 'hard':
        targets = [deepcopy(source) for source in sources]
    elif update_type == 'soft':
        for source, target in zip(sources, targets):
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    else:
        raise ValueError(f'Update type must be "hard" or "soft", not "{update_type}"')

    [target.train(False) for target in targets]
    return targets


def run(logging_path, env_name='CartPole-v1', seed=0, timesteps=5_000_000, trans_per_iter=5000, n_ensemble=2, mode='mean',
        hidden_width=256, kl_weight=20., init_ew=2.0, final_ew=0.4, nb_hidden=2, device='cuda', l2_weight=0.0,
        memory_size=300, udr=1., rwd_scale=10., w_correction=1, rep_mem_size=50_000, n_eval_episodes=25, eval_interval=100_000,
        lr=1e-4, batch_size=256, target_type='hard', hard_target_steps=200, soft_target_polyak=0.005,
        init_eps=0.05, final_eps=0.05, end_decay=500_000):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)
    print('device', device)

    assert target_type in ['hard', 'soft']
    assert mode in ['mean', 'min']

    # preparing env and env_eval
    def make_envs():
        # Init CNN
        network_type = "mlp"
        cnn_config = None
        obs_type = torch.float32
        if env_name.startswith('MinAtar'):
            env = ChannelsFirst(gym.make(env_name))
            env_eval = ChannelsFirst(gym.make(env_name))
            env = TimeLimit(env, max_episode_steps=5000)
            env_eval = TimeLimit(env_eval, max_episode_steps=5000)
            print("Using CNN")
            network_type = "cnn"
            cnn_config = {
                'channels': [16],
                'kernel_size': [3],
                'stride': [1]
            }
        else:
            env = gym.make(env_name)
            env_eval = gym.make(env_name)

        return env, env_eval, network_type, cnn_config, obs_type

    env, env_eval, network_type, cnn_config, obs_type = make_envs()

    gamma = 0.99

    s_dim = env.observation_space.shape if network_type == "cnn" else env.observation_space.shape[0]
    n_act = env.action_space.n
    sampler = Sampler(env)
    sampler_eval = Sampler(env_eval)

    print('logging path', logging_path)
    logger = SummaryWriter(logging_path)

    qfuncs = [StaQNet(s_dim, n_act, nb_hidden=nb_hidden, hidden_width=hidden_width, memory_size=memory_size, use_w_correction=w_correction,
                     kl_weight=kl_weight, entropy_weight=init_ew, device=device, network_type=network_type,
                     cnn_config=cnn_config) for _ in range(n_ensemble)]

    repmem = ReplayMemory(rep_mem_size, s_dim, device, obs_type=obs_type)

    total_trans = 0
    eweight_fct = lambda x: (min(x / end_decay, 1) * final_ew + (1 - min(x / end_decay, 1)) * init_ew) / np.log(n_act)
    qoptim = torch.optim.Adam([p for qfunc in qfuncs for p in qfunc.parameters()], lr=lr, weight_decay=l2_weight)

    qtars = update_target(qfuncs, update_type='hard')

    total_time_elapsed = 0.0
    latest_train_return = None
    latest_eval_return = None
    progress_bar = tqdm(total=timesteps, unit='steps')

    while total_trans < timesteps:
        total_start_time = time()
        eweight = eweight_fct(total_trans)
        [qfunc.set_entropy_weight(eweight) for qfunc in qfuncs]
        logger.add_scalar('pars/entrop_weight', eweight, total_trans)

        # sampling new trans
        [qfunc.train(False) for qfunc in qfuncs]

        eps = (min(total_trans / end_decay, 1) * final_eps + (1 - min(total_trans / end_decay, 1)) * init_eps)
        rollout_start_time = time()

        new_trans, returns, entropies = sampler.rollouts(lambda x: numpy_egreedy_softpolicy(x, qfuncs, eps, device),
                                                         min_trans=trans_per_iter, max_trans=trans_per_iter)

        # logging returns
        for ret, entr in zip(returns, entropies):
            latest_train_return = ret.value
            logger.add_scalar('train/return', ret.value, ret.global_step)
            logger.add_scalar('train/return_n_entropy', ret.value + eweight * entr, ret.global_step)

        repmem.add_trans(new_trans)
        total_trans += trans_per_iter

        rollout_time_elapsed = time() - rollout_start_time

        testb = repmem.sample(200, device=device)  # for logging
        with torch.no_grad():
            # for logging
            old_dist = torch.distributions.Categorical(logits=get_logits_ensemble_torch(qfuncs, testb.obs))
            old_logits_tilde = get_logits_ensemble_torch(qfuncs, testb.obs, no_old=True)
            pol_entropy = old_dist.entropy().mean()

            training_start_time = time()
            precompute_start_time = time()

            # pre-computations
            nologits = torch.zeros(repmem.size, n_act, device=device)
            sid = 0
            for k in range(min(200, repmem.size), repmem.size + 1, 200):
                nologits[sid:k] = get_logits_ensemble_torch(qfuncs, repmem.repmem.nobs[sid:k].to(torch.float32))
                sid = k
            if repmem.size > sid:
                nologits[sid:repmem.size] = get_logits_ensemble_torch(qfuncs, repmem.repmem.nobs[sid:repmem.size].to(torch.float32))

        precompute_time_elapsed = time() - precompute_start_time

        max_q_ratio = 0
        grad_steps = 0
        [qfunc.train(True) for qfunc in qfuncs]

        while grad_steps < int(udr * trans_per_iter):
            if target_type == 'hard' and grad_steps % hard_target_steps == 0:
                qtars = update_target(qfuncs, update_type='hard')
            elif target_type == 'soft':
                qtars = update_target(qfuncs, qtars, soft_target_polyak, update_type='soft')

            qoptim.zero_grad()
            db, idxs = repmem.sample_with_idxs(batch_size, device=device)
            curr_qalls = [qfunc(db.obs) for qfunc in qfuncs]

            with torch.no_grad():
                next_qalls = [qtar(db.nobs) for qtar in qtars]

            max_ent = init_ew * np.log(n_act)
            max_abs_q = (rwd_scale * sampler.max_abs_reward + gamma * max_ent) / (1.0 - gamma)

            # For logging
            current_max_q_ratio = max([q.abs().max().item() for q in curr_qalls] + [q.abs().max().item() for q in next_qalls]) / max_abs_q
            max_q_ratio = max(max_q_ratio, current_max_q_ratio)

            with torch.no_grad():
                # sample next action
                nol = nologits[idxs]
                nopol = torch.distributions.Categorical(logits=nol)
                ## ----- Getting V(no) ------
                na = nopol.sample().view(-1, 1)
                qnos = [qall.gather(dim=1, index=na) for qall in next_qalls]

                ent_term = eweight * nopol.entropy()[:, None]

            scaled_rwd = rwd_scale * db.rwd
            if mode == 'mean':
                targ = sum([qno.detach() for qno in qnos]) / n_ensemble
                targ = scaled_rwd + gamma * (1 - db.terminated) * (targ + ent_term)
                lossq = sum([(qall.gather(dim=1, index=db.act) - targ).pow(2).mean()
                             for qall in curr_qalls]) / n_ensemble
            else:
                targ = torch.hstack([qno.detach() for qno in qnos]).min(1, True)[0]
                targ = scaled_rwd + gamma * (1 - db.terminated) * (targ + ent_term)
                lossq = sum([(qall.gather(dim=1, index=db.act) - targ).pow(2).mean()
                             for qall in curr_qalls]) / n_ensemble

            loss_logqdist = (((eweight * nol - sum(next_qalls) / n_ensemble) ** 2) * (1 - db.terminated)).mean()
            lossq.backward()

            if grad_steps % 100 == 0:
                logger.add_scalar('loss/bellerror', lossq.item(), int(udr * (total_trans - trans_per_iter)) + grad_steps)
                logger.add_scalar('loss/logqdist', loss_logqdist.item(), int(udr * (total_trans - trans_per_iter)) + grad_steps)
                logger.add_scalar('loss/nb_prob_act_u0.01', (nopol.probs < 0.01).sum() / batch_size, int(udr * (total_trans - trans_per_iter)) + grad_steps)
                logger.add_scalar('loss/max_q_ratio', max_q_ratio, int(udr * (total_trans - trans_per_iter)) + grad_steps)

                max_q_ratio = 0

            qoptim.step()
            grad_steps += 1

        # adding weights of current q to sumq
        [qfunc.train(False) for qfunc in qfuncs]
        with torch.no_grad():
            if w_correction:  # for logging only
                old_logits_tilde += (sum([qfunc(testb.obs) for qfunc in qfuncs]) / n_ensemble) * qfuncs[0].eta * qfuncs[0].decay ** (memory_size - 1) / (
                            1 - qfuncs[0].decay ** memory_size)

            [qfunc.update_sigq() for qfunc in qfuncs]
            sumqnext = get_logits_ensemble_torch(qfuncs, testb.obs)

            training_time_elapsed = time() - training_start_time

            # logging
            # doesn't take into account greedy
            logger.add_scalar('log/kl_pk_pkpone', stable_kl_div(old_dist.probs, torch.softmax(sumqnext, dim=1)).mean(), total_trans // trans_per_iter)
            logger.add_scalar('log/kl_pk_pktilde', stable_kl_div(old_dist.probs, torch.softmax(old_logits_tilde, dim=1)).mean(), total_trans // trans_per_iter)
            logger.add_scalar('log/entropy', pol_entropy, total_trans // trans_per_iter)

            # evaluate policy
            if total_trans % eval_interval == 0:
                [qfunc.train(False) for qfunc in qfuncs]
                eval_start_time = time()
                eval_rollouts = [sampler_eval.rollouts(lambda x: numpy_argmax_policy(x, qfuncs, device), 1, np.inf, returns_only=True) for _ in range(n_eval_episodes)]
                latest_eval_return = np.mean([r[1][0].value for r in eval_rollouts])
                logger.add_scalar('eval/return', latest_eval_return, total_trans)
                logger.add_scalar('eval/mean_ep_length', np.mean([r[1][0].step for r in eval_rollouts]), total_trans)
                logger.add_scalar('timings/eval', time() - eval_start_time, total_trans)

        total_time_elapsed += time() - total_start_time

        logger.add_scalar("timings/rollout", rollout_time_elapsed, total_trans)
        logger.add_scalar("timings/precompute", precompute_time_elapsed, total_trans)
        logger.add_scalar("timings/training", training_time_elapsed, total_trans)
        logger.add_scalar("timings/total", total_time_elapsed, total_trans)
        logger.add_scalar('loss/max_abs_reward', sampler.max_abs_reward, total_trans)

        progress_bar.set_postfix({
            'train/return': latest_train_return,
            'eval/return': latest_eval_return,
        }, refresh=False)
        progress_bar.update(trans_per_iter)

    progress_bar.close()

if __name__ == '__main__':
    argp = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argp.add_argument('--experiment', type=str, default='StaQ', help='Experiment name for TensorBoard/WandB.')
    argp.add_argument('--env-name', type=str, default='CartPole-v1', help='Gymnasium env id.')
    argp.add_argument('--seed', type=int, default=0)
    argp.add_argument('--logging-dir', type=str, default='', help='TensorBoard root directory')
    argp.add_argument('--timesteps', type=int, default=5_000_000, help='Total environment steps.')
    argp.add_argument('--memory-size', type=int, default=300, help='Maximum number of Q-functions $M$ retained in the StaQ.')
    argp.add_argument('--n-ensemble', type=int, default=2, help='Number of Q-functions in the ensemble.')
    argp.add_argument('--w-correction', type=int, default=1, help='Whether to use finite-memory weight correction term.')
    argp.add_argument('--trans-per-iter', type=int, default=5000, help='Env steps policy iteration.')
    argp.add_argument('--rep-mem-size', type=int, default=50_000, help='Replay buffer size.')
    argp.add_argument('--device', type=str, default='cuda', help='PyTorch device.')
    argp.add_argument('--mode', type=str, choices=('mean', 'min'), default='mean', help='Aggregate min or mean of ensemble targets.')
    argp.add_argument('--init-ew', type=float, default=2.0, help='Initial entropy weight.')
    argp.add_argument('--final-ew', type=float, default=0.4, help='Final entropy weight.')
    argp.add_argument('--end-decay', type=float, default=500_000, help='Anneal steps for entropy and epsilon.')
    argp.add_argument('--lr', type=float, default=1e-4, help='Adam learning rate.')
    argp.add_argument('--batch-size', type=int, default=256, help='Minibatch size.')
    argp.add_argument('--nb-hidden', type=int, default=2, help='Number of hidden layers.')
    argp.add_argument('--hidden-width', type=int, default=256, help='Hidden layer width.')
    argp.add_argument('--init-eps', type=float, default=0.05, help='Initial epsilon-softmax value.')
    argp.add_argument('--final-eps', type=float, default=0.05, help='Final epsilon-softmax value.')
    argp.add_argument('--l2-weight', type=float, default=0.0, help='Adam weight decay.')
    argp.add_argument('--kl-weight', type=float, default=20., help='KL weight.')
    argp.add_argument('--rwd-scale', type=float, default=10., help='Reward scale.')
    argp.add_argument('--udr', type=float, default=1., help='Update-to-data ratio')
    argp.add_argument('--n-eval-episodes', type=int, default=25, help='Num episodes for policy benchmark.')
    argp.add_argument('--eval-interval', type=int, default=100_000, help='Policy benchmark frequency.')
    argp.add_argument('--target-type', type=str, choices=('hard', 'soft'), default='hard', help='Target network update type.')
    argp.add_argument('--hard-target-steps', type=int, default=200, help='Steps between hard target network updates.')
    argp.add_argument('--soft-target-polyak', type=float, default=0.005, help='Polyak coefficient for soft target-network updates.')
    argp.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True, help='Use --no-wandb to disable WandB.')
    argp.add_argument('--wandb-project', type=str, default='StaQ', help='WandB project name.')

    args = argp.parse_args()

    logging_dir = args.logging_dir
    paras = args.__dict__.copy()
    if logging_dir == '':
        logging_dir = f'logs/{time()}'

    run_name = join(args.experiment, args.env_name, str(args.seed))

    paras['logging_path'] = join(logging_dir, run_name)

    del paras['logging_dir']
    del paras['experiment']

    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            config=paras,
            sync_tensorboard=True,
            name=run_name,
            monitor_gym=True,
        )

    del paras['wandb']
    del paras['wandb_project']

    run(**paras)
