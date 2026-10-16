# StaQ: a Finite Memory Approach to Discrete Action Policy Mirror Descent

Official code for the paper accepted at the **Reinforcement Learning Conference (RLC), 2026**.

_Alex Davey, Alena Shilova, Brahim Driss, Riad Akrour_

**Paper:** [arXiv](https://arxiv.org/abs/2506.13862)

<p align="center">
  <img src="assets/staq-overview.png" alt="Overview of the StaQ policy evaluation and stacked policy update" width="50%"/>
</p>


StaQ is a finite-memory approach to Policy Mirror Descent (PMD) for discrete action spaces. It retains the last $M$ Q-functions in a stacked neural network, giving an optimization-free policy update that retains PMD's error-averaging effect. In the paper, increasing $M$ improves performance up to a point where finite-memory StaQ closely matches exact PMD.

<p align="center">
  <img src="assets/staq_deeprl.png" alt="MinAtar learning curves comparing StaQ with M-DQN, DQN, TRPO, PPO, and NatGrad with line search" width="100%"/>
</p>

## Setup

Install StaQ with either uv or Conda+pip.

### uv

To use uv, simply prefix commands with `uv run` (e.g. `uv run run/staq_classic.sh CartPole-v1 12345`).

### Installing with Conda + pip

If you use uv, you can skip straight to the run instructions. To use Conda+pip, first create a Python 3.10 Conda environment and install the project dependencies:

```bash
conda create -n staq python=3.10
conda activate staq
conda install swig
pip install -r requirements.txt
```

## Running Experiments

The `run/` directory contains the configurations used for the paper. Each command below takes an environment and seed, and is run from the repository root. For example:

```bash
# Classic control
run/staq_classic.sh CartPole-v1 12345

# MinAtar
run/staq_minatar.sh MinAtar/Asterix-v1 12345
```

These paper scripts run for 5M timesteps. More arguments are available in `staq.py`, for example, the StaQ memory size $M$ is set by the `staq.py` flag `--memory-size` (defaults $M=300$). The environments benchmarked in the paper are:

```
CartPole-v1
Acrobot-v1
LunarLander-v2
MountainCar-v0

MinAtar/Asterix-v1
MinAtar/Breakout-v1
MinAtar/Freeway-v1
MinAtar/Seaquest-v1
MinAtar/SpaceInvaders-v1
```


## Baselines

The Actor-Critic baselines are given in `baselines/`. For Conda + pip, install the extra dependencies with

```bash
pip install -r requirements_baselines.txt
```

Then baselines can be run with e.g. (for `NatGrad + LS`)

```bash
run/baselines/natgrad_ls_trpo_minatar.sh MinAtar/Asterix-v1 12345
```

For uv, prefix runs with `uv run --extra baselines`, e.g. `uv run --extra baselines run/baselines/natgrad_ls_trpo_minatar.sh MinAtar/Asterix-v1 12345`.

The baselines supported are:

| Script | Baseline | Entry point |
|---|---|---|
| `natgrad_ls_trpo_minatar.sh` | NatGrad + LS: natural gradient with a KL line search | `baselines/nat_grad.py` |
| `ac_directopt_mdpo_minatar.sh` | MDPO: gradient steps on the KL-regularized policy objective | `baselines/mdpo_ecpo_sac.py --pupdate-type MDPO` |
| `ac_mproj_ecpo_minatar.sh` | ECPO: projection onto the closed-form KL-regularized policy | `baselines/mdpo_ecpo_sac.py --pupdate-type ECPO` |
| `ac_nokl_sac_minatar.sh` | SAC: MDPO without the KL term | `baselines/mdpo_ecpo_sac.py --pupdate-type SAC` |

## Citations

To cite the paper and/or this repository:

```bibtex
@article{davey2026staq,
    title={StaQ: a Finite Memory Approach to Discrete Action Policy Mirror Descent},
    author={Alex Davey and Alena Shilova and Brahim Driss and Riad Akrour},
    journal={Reinforcement Learning Journal},
    volume={7},
    pages={},
    year={2026}
}
```
