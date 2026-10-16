from copy import deepcopy
import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from time import time
import argparse
from os.path import join
from gymnasium.wrappers import TimeLimit
from baseline_tools import ChannelsFirst, stable_kl_div, aggregate, make_neural_model, StackedEnsemble, SamplerWithLogits, ReplayMemoryWithLogits
from cg_metric_grad_sb import MetricGradPO
from tqdm import tqdm


class SoftMaxPolicy(nn.Module):
    def __init__(self, nn, nb_act):
        super().__init__()
        self.nn = nn
        self.nb_act = nb_act

    def get_logits(self, obs):
        return self.nn(obs)


def get_logits(policy, obs, device, obs_preprocess):
    return policy.get_logits(obs_preprocess(torch.as_tensor(np.asarray(obs)[None, :], device=device)))

def numpy_softmax(obs, policy, device, obs_preprocess):
    with torch.no_grad():
        logits = get_logits(policy, obs, device, obs_preprocess)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().squeeze().cpu().numpy(), dist.entropy().squeeze().cpu().numpy(), logits.squeeze().cpu().numpy()


def numpy_argmax_policy(obs, policy, device, obs_preprocess):
    with torch.no_grad():
        logits = get_logits(policy, obs, device, obs_preprocess)
        dist = torch.distributions.Categorical(logits=logits)
    return dist.probs.argmax(1).squeeze().cpu().numpy(), dist.entropy().squeeze().cpu().numpy(), logits.squeeze().cpu().numpy()


def numpy_egreedy_softpolicy(obs, policy, eps, device, obs_preprocess):
    with torch.no_grad():
        act, entrop, logits = numpy_softmax(obs, policy, device, obs_preprocess)
    if np.random.rand() < eps:
        return np.random.randint(policy.nb_act), entrop, logits
    return act, entrop, logits


def default_obs_preprocess(x):
    return x.to(torch.float32)

def make_envs(env_name):
    network_config = dict()
    obs_type = torch.float32
    obs_pre_process = default_obs_preprocess
    env = ChannelsFirst(gym.make(env_name))
    env_eval = ChannelsFirst(gym.make(env_name, render_mode='rgb_array'))
    env = TimeLimit(env, max_episode_steps=5000)
    env_eval = TimeLimit(env_eval, max_episode_steps=5000)
    network_config['cnn_config'] = {
        'channels': [16],
        'kernel_size': [3],
        'stride': [1]
    }
    return env, env_eval, network_config, obs_type, obs_pre_process


def run(logging_path, env_name='MinAtar/Asterix-v1', seed=0, timesteps=5_000_000, trans_per_iter=5000, n_ensemble=2, mode='mean',
        hidden_width=128, kl_weight=20., init_ew=2.0, final_ew=0.4, device='cuda', l2_weight=0.0,
        max_expand=300, udr=1., rwd_scale=100., rep_mem_size=50_000, n_eval_episodes=25, eval_interval=200_000,
        lr=1e-4, batch_size=256, hard_target_steps=200,
        init_eps=0.05, final_eps=0.05, end_decay=1_000_000, device_mem='cuda', max_kl=0.001,
        pupdate_batch_size=1000, use_wandb=False, fast=True, verbose=False):
    if use_wandb:
        import wandb

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if fast:
        torch.distributions.categorical.Distribution.set_default_validate_args(False)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if not torch.cuda.is_available():
        device = "cpu"
        device_mem = "cpu"
    device_str = device
    device = torch.device(device)
    print('device', device)
    device_mem_str = device_mem
    device_mem = torch.device(device_mem)


    # preparing env and env_eval
    env, env_eval, network_config, obs_type, obs_preprocess = make_envs(env_name)

    gamma = 0.99

    s_dim = env.observation_space.shape
    n_act = env.action_space.n

    # Distinct seeds for the training and evaluation environments, derived from the run's seed
    train_seed, eval_seed = np.random.SeedSequence(seed).generate_state(2).tolist()
    sampler_eval = SamplerWithLogits(env_eval, seed=eval_seed)
    sampler = SamplerWithLogits(env, seed=train_seed)

    print('logging path', logging_path)
    logger = SummaryWriter(logging_path)

    network_config['input_size'] = s_dim
    network_config['output_size'] = n_act
    network_config['nb_neurones'] = hidden_width

    strides = network_config['cnn_config']['stride']

    qenble = StackedEnsemble([make_neural_model(network_config) for _ in range(n_ensemble)], strides=strides).to(device)
    policy = SoftMaxPolicy(nn=make_neural_model(network_config).to(device), nb_act=n_act)
    poptim = MetricGradPO(policy, lr=1. / kl_weight, max_step=None if max_kl == 0.0 else max_kl,
                          verbose=verbose)

    repmem = ReplayMemoryWithLogits(rep_mem_size, s_dim, n_act, device_mem, obs_type=obs_type)

    total_trans = 0
    total_grad_steps = 0
    eweight_fct = lambda x: (min(x / end_decay, 1) * final_ew + (1 - min(x / end_decay, 1)) * init_ew) / np.log(n_act)
    qoptim = torch.optim.Adam(qenble.parameters(), lr=lr, weight_decay=l2_weight)

    qtars = deepcopy(qenble)

    total_time_elapsed = 0.0

    latest_train_return = None
    latest_eval_return = None
    progress_bar = tqdm(total=timesteps, unit='steps')

    while total_trans < timesteps:
        total_start_time = time()
        eweight = eweight_fct(total_trans)
        logger.add_scalar('pars/entrop_weight', eweight, total_trans)
        eps = (min(total_trans / end_decay, 1) * final_eps + (1 - min(total_trans / end_decay, 1)) * init_eps)
        rollout_start_time = time()

        new_trans, returns, entropies = sampler.rollouts(lambda x: numpy_egreedy_softpolicy(x, policy, eps, device, obs_preprocess), min_trans=trans_per_iter, max_trans=trans_per_iter)

        # logging returns
        for ret, entr in zip(returns, entropies):
            latest_train_return = ret.value
            logger.add_scalar('train/ep_length', ret.step, ret.global_step)
            logger.add_scalar('train/return', ret.value, ret.global_step)
            logger.add_scalar('train/return_n_entropy', ret.value + eweight * entr, ret.global_step)

        repmem.add_trans(new_trans)
        total_trans += trans_per_iter

        rollout_time_elapsed = time() - rollout_start_time

        testb = repmem.sample(200, device=device)  # for logging
        with torch.no_grad():
            # for logging
            old_dist = torch.distributions.Categorical(logits=policy.get_logits(obs_preprocess(testb.obs)))
            pol_entropy = old_dist.entropy().mean()

            precompute_start_time = training_start_time = time()

            if total_trans > trans_per_iter and rep_mem_size > trans_per_iter:
                # pre-computations on second iter and onward
                ndataprec = repmem.size - trans_per_iter
                nologits = repmem.get('nologits', start_idx=0, end_idx=ndataprec).to(device)
                batch_precomp = 2000
                for k in range(0, ndataprec, batch_precomp):
                    lidx, ridx = k, min(k + batch_precomp, ndataprec)
                    if device_mem_str != 'cpu' or device_str == 'cpu':
                        nobs = repmem.get('nobs', start_idx=lidx, end_idx=ridx)  # not done for all replay buffer as with nologits to save on memory if obs are heavy (e.g. images)
                    else:
                        nobs = repmem.get('nobs', start_idx=lidx, end_idx=ridx).pin_memory()  # not done for all replay buffer as with nologits to save on memory if obs are heavy (e.g. images)
                    nologits[lidx:ridx] = policy.get_logits(obs_preprocess(nobs.to(device=device)))
                repmem.set('nologits', nologits, start_idx=0, end_idx=ndataprec)

        precompute_time_elapsed = time() - precompute_start_time

        max_q_ratio = 0

        # --------- policy eval ------------
        if device_mem_str != 'cpu' or device_str == 'cpu':
            dataloader = DataLoader(repmem, batch_size=batch_size, shuffle=True, drop_last=True)
        else:
            dataloader = DataLoader(repmem, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True, pin_memory_device=device_str)
        grad_steps = 0
        while grad_steps < int(udr * trans_per_iter):  # utd = 1
            for data in dataloader:
                grad_steps += 1
                total_grad_steps += 1
                obs, act, rwd, terminated, nobs, nologits = obs_preprocess(data[0].to(device)), data[1].to(device), data[2].to(device), data[3].to(device), obs_preprocess(data[4].to(device)), data[5].to(device)
                # Update target networks (if using)
                if total_grad_steps % hard_target_steps == 0:
                    qtars = deepcopy(qenble)

                qoptim.zero_grad()

                # Use either Cross-Q or Fitted-Q
                curr_qalls = qenble(obs)

                with torch.no_grad():
                    next_q = aggregate(qtars(nobs), mode)

                with torch.no_grad():
                    # sample next action
                    nopol = torch.distributions.Categorical(logits=nologits)

                    ## ----- Getting V(no) ------
                    na = nopol.sample().view(-1, 1)
                    qnos = next_q.gather(dim=1, index=na)

                    ent_term = eweight * nopol.entropy()[:, None]

                targ = rwd_scale * rwd + gamma * (1 - terminated) * (qnos + ent_term)
                lossq = sum([(curr_q.gather(dim=-1, index=act) - targ).pow(2).mean() for curr_q in curr_qalls]) / n_ensemble

                loss_logqdist = ((eweight * nologits - next_q).pow(2) * (1 - terminated)).mean()
                lossq.backward()

                if grad_steps % 100 == 0:
                    logger.add_scalar('loss/bellerror', lossq.item(), int(udr * (total_trans - trans_per_iter)) + grad_steps)
                    logger.add_scalar('loss/logqdist', loss_logqdist.item(), int(udr * (total_trans - trans_per_iter)) + grad_steps)
                    logger.add_scalar('loss/nb_prob_act_u0.01', (nopol.probs < 0.01).sum() / batch_size, int(udr * (total_trans - trans_per_iter)) + grad_steps)
                    logger.add_scalar('loss/max_q_ratio', max_q_ratio, int(udr * (total_trans - trans_per_iter)) + grad_steps)

                if grad_steps % 1000 == 0 and use_wandb:
                    wandb.log({'loss/next_obs_probs': wandb.Histogram(nopol.probs.flatten().cpu().numpy())})

                    max_q_ratio = 0

                qoptim.step()
                if grad_steps >= int(udr * trans_per_iter):
                    break

        # ------- policy update -----------
        pup_batch = repmem.sample(pupdate_batch_size, device=device)
        if max_kl > 0.0:
            with torch.no_grad():
                old_dist_forup = torch.distributions.Categorical(logits=policy.get_logits(obs_preprocess(pup_batch.obs)))

        def objective_n_kl(pol, kl_to_old=False):
            if kl_to_old:
                with torch.no_grad():
                    dist = torch.distributions.Categorical(logits=pol.get_logits(obs_preprocess(pup_batch.obs)))
                    return  nn.functional.kl_div(old_dist_forup.logits, dist.logits, log_target=True)
            dist = torch.distributions.Categorical(logits=pol.get_logits(obs_preprocess(pup_batch.obs)))
            qvals_pup = aggregate(qenble(obs_preprocess(pup_batch.obs)), mode)
            return (qvals_pup * dist.probs).sum(1).mean() + eweight * dist.entropy().mean(), nn.functional.kl_div(dist.logits.detach(), dist.logits, log_target=True)

        poptim.step(objective_n_kl)


        with torch.no_grad():
            training_time_elapsed = time() - training_start_time

            sumqnext = policy.get_logits(obs_preprocess(testb.obs))


            # logging
            # doesn't take into account greedy
            logger.add_scalar('log/kl_pk_pkpone', stable_kl_div(old_dist.probs, torch.softmax(sumqnext, dim=1)).mean(), total_trans // trans_per_iter)
            logger.add_scalar('log/entropy', pol_entropy, total_trans // trans_per_iter)

            # evaluate policy
            if total_trans % eval_interval == 0:
                eval_start_time = time()
                eval_rollouts = [sampler_eval.rollouts(lambda x: numpy_argmax_policy(x, policy, device, obs_preprocess), 1, np.inf, returns_only=True) for _ in range(n_eval_episodes)]
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
        logger.add_scalar('loss/rwd_scale', rwd_scale, total_trans)

        progress_bar.set_postfix({
            'train/return': latest_train_return,
            'eval/return': latest_eval_return,
        }, refresh=False)
        progress_bar.update(trans_per_iter)

    progress_bar.close()
    logger.close()

if __name__ == '__main__':
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    argp = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argp.add_argument('--env-name', type=str, default='MinAtar/Asterix-v1', help='MinAtar Gymnasium environment id.')
    argp.add_argument('--experiment', type=str, default='NatGrad', help='Experiment name, the first part of the run name <experiment>/<env>/<seed>.')
    argp.add_argument('--seed', type=int, default=0, help='Random seed.')
    argp.add_argument('--logging-dir', type=str, default='', help='TensorBoard root directory (logs/<timestamp> if empty).')
    argp.add_argument('--timesteps', type=int, default=5_000_000, help='Total environment steps.')
    argp.add_argument('--n-ensemble', type=int, default=2, help='Number of Q-functions in the ensemble.')
    argp.add_argument('--trans-per-iter', type=int, default=5000, help='Environment steps per iteration.')
    argp.add_argument('--rep-mem-size', type=int, default=50_000, help='Replay buffer size.')
    argp.add_argument('--device', type=str, default='cuda', help='PyTorch device; falls back to CPU if CUDA is unavailable.')
    argp.add_argument('--device-mem', type=str, default='cuda', help='Device of the replay buffer; falls back to CPU if CUDA is unavailable.')
    argp.add_argument('--mode', type=str, choices=('mean', 'min'), default='mean', help='Aggregation of the ensemble Q-values (Q target and policy update).')
    argp.add_argument('--init-ew', type=float, default=2.0, help='Initial entropy weight (divided by log of the number of actions).')
    argp.add_argument('--final-ew', type=float, default=0.4, help='Final entropy weight (divided by log of the number of actions).')
    argp.add_argument('--max-kl', type=float, default=0.001, help='KL bound of the line search on the natural gradient step (0 disables the line search).')
    argp.add_argument('--pupdate-batch-size', type=int, default=1000, help='Transitions sampled for each policy update.')
    argp.add_argument('--end-decay', type=float, default=1_000_000, help='Environment steps over which the entropy weight and epsilon are annealed.')
    argp.add_argument('--lr', type=float, default=1e-4, help='Adam learning rate of the Q-functions.')
    argp.add_argument('--batch-size', type=int, default=256, help='Minibatch size of the Q-function updates.')
    argp.add_argument('--hidden-width', type=int, default=128, help='Hidden layer width of the policy and Q-functions.')
    argp.add_argument('--init-eps', type=float, default=0.05, help='Initial epsilon of the epsilon-greedy exploration.')
    argp.add_argument('--final-eps', type=float, default=0.05, help='Final epsilon of the epsilon-greedy exploration.')
    argp.add_argument('--l2-weight', type=float, default=0.0, help='Adam weight decay of the Q-functions.')
    argp.add_argument('--kl-weight', type=float, default=20., help='KL weight; the natural gradient step size is 1 / kl-weight.')
    argp.add_argument('--rwd-scale', type=float, default=100., help='Reward scale.')
    argp.add_argument('--udr', type=float, default=1., help='Update-to-data ratio of the Q-functions.')
    argp.add_argument('--n-eval-episodes', type=int, default=25, help='Episodes per policy evaluation.')
    argp.add_argument('--eval-interval', type=int, default=200_000, help='Environment steps between policy evaluations.')
    argp.add_argument('--hard-target-steps', type=int, default=200, help='Gradient steps between hard target network updates.')
    argp.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True, help='Use --no-wandb to disable WandB.')
    argp.add_argument('--wandb-project', type=str, default='StaQ', help='WandB project name.')
    argp.add_argument('--fast', action=argparse.BooleanOptionalAction, default=True, help='Faster, but potentially nondeterministic, PyTorch settings.')
    argp.add_argument('--verbose', action=argparse.BooleanOptionalAction, default=False, help='Print the KL at each line-search step.')

    args = argp.parse_args()

    logging_dir = args.logging_dir
    paras = args.__dict__.copy()
    if logging_dir == '':
        logging_dir = f'logs/{time()}'

    run_name = join(args.experiment, args.env_name, str(args.seed))

    paras['logging_path'] = join(logging_dir, run_name)

    del paras['logging_dir']
    del paras['experiment']

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            config=paras,
            sync_tensorboard=True,
            name=run_name,
            monitor_gym=True,
        )

    paras['use_wandb'] = paras['wandb']
    del paras['wandb']
    del paras['wandb_project']

    try:
        run(**paras)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
