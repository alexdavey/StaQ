from copy import deepcopy
import gymnasium as gym
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from time import time
import argparse
from os.path import join
from gymnasium.wrappers import TimeLimit
from baseline_tools import Sampler, ReplayMemory, ChannelsFirst, stable_kl_div, aggregate, make_neural_model, StackedEnsemble
from tqdm import tqdm

def get_logits(actor, obs, device):
    if obs.__class__ is torch.Tensor:
        return actor(obs)
    else:
        return actor(torch.tensor(obs[None, :].astype(np.float32), device=device))


def apply_weight_mask(logits, mode, n_samples, past_qs_weight):
    # Apply weight mask to logits of shape [M, batch, actions]
    if mode == 'weighted_sum':
        mask = torch.tensor(past_qs_weight, device=logits.device)[:, None, None]
    elif mode == 'sample_qs':
        norma_term = sum(past_qs_weight)
        past_qs_probs = np.asarray(past_qs_weight) / norma_term

        qidx = np.random.choice(logits.shape[0], replace=False, p=past_qs_probs, size=n_samples)
        mask = torch.zeros(logits.shape[0], logits.shape[1], 1, device=logits.device)
        mask[qidx] = norma_term / n_samples
    else:
        raise ValueError(f'Unknown masking mode {mode}')

    return (mask * logits).sum(0)


def numpy_egreedy(x, actor, eps, device, soft=True):
    logits = get_logits(actor, x, device)
    pol = torch.distributions.Categorical(logits=logits)
    entropy = pol.entropy().detach().squeeze().cpu().numpy()
    if np.random.rand() < eps:
        return np.random.randint(logits.shape[1]), entropy
    if soft:
        act = pol.sample().squeeze().cpu().numpy()
    else:
        act = logits.argmax(dim=1).squeeze().cpu().numpy()
    return act, entropy


def run(logging_path, env_name='MinAtar/Asterix-v1', seed=0, timesteps=5_000_000, trans_per_iter=5000, n_ensemble=2,
        aggreg_q='mean', hidden_width_pi=128, hidden_width_q=128, kl_weight=20., init_ew=2.0, final_ew=0.4, device='cuda', l2_weight=0.0,
        aggreg_pi='mean', rwd_scale=100., rep_mem_size=50_000, n_eval_episodes=25, eval_interval=200_000,
        lr_q=1e-4, lr_pi=1e-4, batch_size=256, hard_target_steps=200,
        init_eps=0.05, final_eps=0.05, end_decay=1_000_000, pupdate_type='MDPO',
        udr=1.):  # udr only for the policy update, for Q-function use an udr of 1.0
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)
    print('device', device)

    assert pupdate_type in ['MDPO', 'ECPO', 'SAC']

    # preparing env and env_eval
    def make_envs():
        # Init CNN
        network_config = dict()
        obs_type = torch.float32
        env = ChannelsFirst(gym.make(env_name))
        env_eval = ChannelsFirst(gym.make(env_name, render_mode='rgb_array'))
        env = TimeLimit(env, max_episode_steps=5000)
        env_eval = TimeLimit(env_eval, max_episode_steps=5000)
        network_config['cnn_config'] = {
            'channels': [16],
            'kernel_size': [3],
            'stride': [1]
        }

        return env, env_eval, network_config, obs_type

    env, env_eval, network_config, obs_type = make_envs()

    gamma = 0.99

    s_dim = env.observation_space.shape
    n_act = env.action_space.n
    network_config['input_size'] = s_dim
    network_config['output_size'] = n_act

    # Distinct seeds for the training and evaluation environments, derived from the run's seed
    train_seed, eval_seed = np.random.SeedSequence(seed).generate_state(2).tolist()
    sampler = Sampler(env, seed=train_seed)
    sampler_eval = Sampler(env_eval, seed=eval_seed)

    print('logging path', logging_path)
    logger = SummaryWriter(logging_path)

    network_config['nb_neurones'] = hidden_width_pi
    actor = make_neural_model(network_config).to(device)
    network_config['nb_neurones'] = hidden_width_q

    qfuns = [make_neural_model(network_config) for _ in range(n_ensemble)]

    qstack = StackedEnsemble(qfuns, strides=network_config['cnn_config']['stride'])

    qstack = qstack.to(device)

    repmem = ReplayMemory(rep_mem_size, s_dim, device, obs_type=obs_type)

    total_trans = 0
    eweight_fct = lambda x: (min(x / end_decay, 1) * final_ew + (1 - min(x / end_decay, 1)) * init_ew) / np.log(n_act)
    actoptim = torch.optim.Adam(actor.parameters(), lr=lr_pi)

    qoptim = torch.optim.Adam(qstack.parameters(), lr=lr_q, weight_decay=l2_weight)

    qtars = deepcopy(qstack)

    grad_log_freq = 100

    latest_train_return = None
    latest_eval_return = None
    progress_bar = tqdm(total=timesteps, unit='steps')

    while total_trans < timesteps:
        # Sampling new rollouts
        eweight = eweight_fct(total_trans)
        logger.add_scalar('pars/entrop_weight', eweight, total_trans)
        eps = (min(total_trans / end_decay, 1) * final_eps + (1 - min(total_trans / end_decay, 1)) * init_eps)

        new_trans, returns, entropies = sampler.rollouts(lambda x: numpy_egreedy(x, actor, eps, device), min_trans=trans_per_iter, max_trans=trans_per_iter)
        # logging returns
        for ret, entr in zip(returns, entropies):
            latest_train_return = ret.value
            logger.add_scalar('train/return', ret.value, ret.global_step)
            logger.add_scalar('train/return_n_entropy', ret.value + eweight * entr, ret.global_step)
        repmem.add_trans(new_trans)
        total_trans += trans_per_iter

        with torch.no_grad():
            # KL and entropy logging
            testb = repmem.sample(200, device=device)  # for logging
            old_dist = torch.distributions.Categorical(logits=get_logits(actor, testb.obs, device))
            pol_entropy = old_dist.entropy().mean()

            # Pre-computation of next state action probabilities
            nologits = torch.zeros(repmem.size, n_act, device=device)
            batch_precomp = 1000
            for k in range(0, repmem.size, batch_precomp):
                lidx, ridx = k, min(k + batch_precomp, repmem.size)
                nologits[lidx:ridx] = get_logits(actor, repmem.repmem.nobs[lidx:ridx].to(torch.float32), device)

        max_q_ratio = 0

        # Updating Q-function
        udr_q = 1.0
        for grad_steps in range(int(udr_q * trans_per_iter)):
            # Update target networks (if using)
            if grad_steps % hard_target_steps == 0:
                qtars = deepcopy(qstack)

            qoptim.zero_grad()
            db, idxs = repmem.sample_with_idxs(batch_size, device=device)

            curr_qalls = qstack(db.obs)

            with torch.no_grad():
                next_qalls = qtars(db.nobs)

            next_q = aggregate(next_qalls, aggreg_q)

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
                qnos = next_q.gather(dim=-1, index=na)

                ent_term = eweight * nopol.entropy()[:, None]

            targ = rwd_scale * db.rwd + gamma * (1 - db.terminated) * (qnos + ent_term)
            lossq = sum([(curr_q.gather(dim=-1, index=db.act) - targ).pow(2).mean() for curr_q in curr_qalls]) / n_ensemble

            lossq.backward()

            if grad_steps % grad_log_freq == 0:
                q_log_step = int(udr_q * (total_trans - trans_per_iter)) + grad_steps
                logger.add_scalar('loss/bellerror', lossq.item(), q_log_step)
                loss_logqdist = ((((eweight * nol) - (eweight * nol).mean(1, True)) - (next_q - next_q.mean(1, True))) * (1 - db.terminated)).pow(2).mean()
                logger.add_scalar('loss/logqdist', loss_logqdist.item(), q_log_step)
                logger.add_scalar('loss/nb_prob_act_u0.01', (nopol.probs < 0.01).sum() / batch_size, q_log_step)
                logger.add_scalar('loss/max_q_ratio', max_q_ratio, q_log_step)

            qoptim.step()

        # Updating actor network
        with torch.no_grad():
            # Pre-computation of next state action probabilities
            all_oldlogits = torch.zeros(repmem.size, n_act, device=device)
            all_currqs = torch.zeros(repmem.size, n_act, device=device)
            batch_precomp = 1000
            for k in range(0, repmem.size, batch_precomp):
                lidx, ridx = k, min(k + batch_precomp, repmem.size)
                all_oldlogits[lidx:ridx] = get_logits(actor, repmem.repmem.obs[lidx:ridx].to(torch.float32), device)
                qvals = qstack(repmem.repmem.obs[lidx:ridx].to(torch.float32))
                all_currqs[lidx:ridx] = aggregate(qvals, aggreg_pi)

        for grad_steps in range(int(udr * trans_per_iter)):  # utd = 1
            actoptim.zero_grad()
            # compute Q values
            minbatch, idxs = repmem.sample_with_idxs(batch_size, device=device)
            qvals = all_currqs[idxs]
            oldlogits = all_oldlogits[idxs]

            actor_logits = actor(minbatch.obs)
            actor_distrib = torch.distributions.Categorical(logits=actor_logits)

            if pupdate_type == 'MDPO':
                # KL loss
                old_actor_distrib = torch.distributions.Categorical(logits=oldlogits)
                loss = - (actor_distrib.probs * qvals).sum(1).mean() - eweight * actor_distrib.entropy().mean() + kl_weight * stable_kl_div(old_actor_distrib.probs, actor_distrib.probs).sum(1).mean()
            elif pupdate_type == 'SAC':
                # MDPO without the KL term
                loss = - (actor_distrib.probs * qvals).sum(1).mean() - eweight * actor_distrib.entropy().mean()
            elif pupdate_type == 'ECPO':
                alpha = 1. / (eweight + kl_weight)
                beta = kl_weight / (eweight + kl_weight)
                target_dist = torch.distributions.Categorical(logits=beta * oldlogits + alpha * qvals)
                loss = stable_kl_div(actor_distrib.probs, target_dist.probs).sum(1).mean()
            loss.backward()
            actoptim.step()

        # Logging
        with torch.no_grad():

            # logging doesn't take into account greedy
            new_dist = torch.distributions.Categorical(logits=get_logits(actor, testb.obs, device))
            logger.add_scalar('log/kl_pk_pkpone', stable_kl_div(old_dist.probs, new_dist.probs).mean(), total_trans // trans_per_iter)
            logger.add_scalar('log/entropy', pol_entropy, total_trans // trans_per_iter)

            # evaluate policy
            if total_trans % eval_interval == 0:
                eval_rollouts = [sampler_eval.rollouts(lambda x: numpy_egreedy(x, actor, eps=0, device=device, soft=False), 1, np.inf, returns_only=True) for _ in range(n_eval_episodes)]
                latest_eval_return = np.mean([r[1][0].value for r in eval_rollouts])
                logger.add_scalar('eval/return', latest_eval_return, total_trans)
                logger.add_scalar('eval/mean_ep_length', np.mean([r[1][0].step for r in eval_rollouts]), total_trans)

        progress_bar.set_postfix({
            'train/return': latest_train_return,
            'eval/return': latest_eval_return,
        }, refresh=False)
        progress_bar.update(trans_per_iter)

    progress_bar.close()
    logger.close()


if __name__ == '__main__':
    argp = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argp.add_argument('--experiment', type=str, default='', help='Experiment name, the first part of the run name <experiment>/<env>/<seed> (the policy update type if empty).')
    argp.add_argument('--env-name', type=str, default='MinAtar/Asterix-v1', help='MinAtar Gymnasium environment id.')
    argp.add_argument('--seed', type=int, default=0, help='Random seed.')
    argp.add_argument('--logging-dir', type=str, default='', help='TensorBoard root directory (logs/<timestamp> if empty).')
    argp.add_argument('--timesteps', type=int, default=5_000_000, help='Total environment steps.')
    argp.add_argument('--n-ensemble', type=int, default=2, help='Number of Q-functions in the ensemble.')
    argp.add_argument('--trans-per-iter', type=int, default=5000, help='Environment steps per iteration.')
    argp.add_argument('--rep-mem-size', type=int, default=50_000, help='Replay buffer size.')
    argp.add_argument('--device', type=str, default='cuda', help='PyTorch device; falls back to CPU if CUDA is unavailable.')
    argp.add_argument('--aggreg-q', type=str, choices=('mean', 'min'), default='mean', help='Aggregation of the ensemble next-state Q-values in the Q target.')
    argp.add_argument('--aggreg-pi', type=str, choices=('mean', 'min'), default='mean', help='Aggregation of the ensemble Q-values in the policy update.')
    argp.add_argument('--init-ew', type=float, default=2.0, help='Initial entropy weight (divided by log of the number of actions).')
    argp.add_argument('--final-ew', type=float, default=0.4, help='Final entropy weight (divided by log of the number of actions).')
    argp.add_argument('--end-decay', type=float, default=1_000_000, help='Environment steps over which the entropy weight and epsilon are annealed.')
    argp.add_argument('--lr-q', type=float, default=1e-4, help='Adam learning rate of the Q-functions.')
    argp.add_argument('--lr-pi', type=float, default=1e-4, help='Adam learning rate of the actor.')
    argp.add_argument('--batch-size', type=int, default=256, help='Minibatch size.')
    argp.add_argument('--hidden-width-q', type=int, default=128, help='Hidden layer width of the Q-functions.')
    argp.add_argument('--hidden-width-pi', type=int, default=128, help='Hidden layer width of the actor.')
    argp.add_argument('--init-eps', type=float, default=0.05, help='Initial epsilon of the epsilon-greedy exploration.')
    argp.add_argument('--final-eps', type=float, default=0.05, help='Final epsilon of the epsilon-greedy exploration.')
    argp.add_argument('--l2-weight', type=float, default=0.0, help='Adam weight decay of the Q-functions.')
    argp.add_argument('--kl-weight', type=float, default=20., help='KL weight of the policy update (unused by SAC).')
    argp.add_argument('--rwd-scale', type=float, default=100., help='Reward scale.')
    argp.add_argument('--udr', type=float, default=1., help='Update-to-data ratio of the actor (the Q-functions always use 1).')
    argp.add_argument('--n-eval-episodes', type=int, default=25, help='Episodes per policy evaluation.')
    argp.add_argument('--eval-interval', type=int, default=200_000, help='Environment steps between policy evaluations.')
    argp.add_argument('--pupdate-type', type=str, choices=('MDPO', 'ECPO', 'SAC'), default='MDPO', help='Policy update: MDPO (direct optimisation), ECPO (projection) or SAC (MDPO without the KL term).')
    argp.add_argument('--hard-target-steps', type=int, default=200, help='Gradient steps between hard target network updates.')
    argp.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True, help='Use --no-wandb to disable WandB.')
    argp.add_argument('--wandb-project', type=str, default='StaQ', help='WandB project name.')

    args = argp.parse_args()

    logging_dir = args.logging_dir
    paras = args.__dict__.copy()
    if logging_dir == '':
        logging_dir = f'logs/{time()}'

    experiment = args.experiment if args.experiment != '' else args.pupdate_type
    run_name = join(experiment, args.env_name, str(args.seed))

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

    del paras['wandb']
    del paras['wandb_project']

    try:
        run(**paras)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
