from __future__ import absolute_import, division, print_function

import argparse
import os

import warnings ; warnings.filterwarnings('ignore')
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import os
import random
import tempfile
from argparse import RawTextHelpFormatter

import gfootball.env as football_env
import gym
import numpy as np
import ray
import torch
from gfootball import env as fe
from gym import wrappers
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler
from rldm.utils import football_tools as ft
from rldm.utils import gif_tools as gt
from rldm.utils import system_tools as st
from rldm.utils.collection_tools import deep_merge
from ray.rllib.agents import with_common_config
from ray.rllib.utils.typing import TrainerConfigDict
# from ray.rllib.agents import COMMON_CONFIG
from ray.rllib.models import ModelCatalog
from ray.rllib.examples.models.centralized_critic_models import (
    CentralizedCriticModel,
    TorchCentralizedCriticModel,
)

EXAMPLE_USAGE = """
Example usage:

    Train one sample of independent policies
        python -m rldm.scripts.train_agents

    Train four samples of independent policies sequentially
        python -m rldm.scripts.train_agents -n 4

    Train four samples of independent policies, two at a time
        python -m rldm.scripts.train_agents -n 4 -m 2

    Train four samples of independent policies, all at once
        python -m rldm.scripts.train_agents -n 4 -o

    Train 50 samples of independent policies, using an ASHA scheduler
        python -m rldm.scripts.train_agents -n 50 -a

    Train one sample of independent policies, enable debug mode
        python -m rldm.scripts.train_agents -d

    To play around with the arguments and not start any training runs:
        python -m rldm.scripts.train_agents -r # add parameters
        For example:
            python -m rldm.scripts.train_agents -r -n 4 -o -d
            python -m rldm.scripts.train_agents -r -o -d -g 0

Checkpoints provided were trained with:

    python -m rldm.scripts.train_agents -n 100 -m 4 -t 50000000 -a -e -b
        This schedules 100 samples, 4 at a time, each up to 50M steps,
        but using the ASHA scheduler, and with callbacks collecting metrics

We recommend you start using all of your resources for a single sample,
using the hyperparameters provided:

    python -m rldm.scripts.train_agents -b -t 20000000

"""


def main(n_cpus, n_gpus, env_name,
         n_policies, n_timesteps, n_samples,
         sample_cpu, sample_gpu, use_scheduler,
         use_tune_config, use_callbacks, debug):
    ray.init(num_cpus=n_cpus, num_gpus=n_gpus, local_mode=debug)

    register_env(env_name, lambda _: ft.RllibGFootball(env_name=env_name))
    obs_space, act_space = ft.get_obs_act_space(env_name)

    def gen_policy(idx):
        return (None, obs_space[f'player_{idx}'], act_space[f'player_{idx}'], {})

    policies = {
        'agent_{}'.format(idx): gen_policy(idx) for idx in range(n_policies)
    }
    policy_ids = list(policies.keys())
    assert len(policy_ids) == 1 or n_policies == len(obs_space), \
        "Script expects either shared or independent policies for all players"
    policy_mapping_fn = lambda agent_id, episode, **kwargs: \
        policy_ids[0 if len(policy_ids) == 1 else int(agent_id.split('_')[1])]

    default_multiagent = {
        'policies': policies,
        'policy_mapping_fn': policy_mapping_fn,
    }

    shared_policy = {'agent_0': gen_policy(0)}
    shared_policy_mapping_fn = lambda agent_id, episode, **kwargs: 'agent_0'
    shared_multiagent = {
        'policies': shared_policy,
        'policy_mapping_fn': shared_policy_mapping_fn,
    }
    ModelCatalog.register_custom_model(
        "cc_model",
        # TorchCentralizedCriticModel
        # if args.framework == "torch"
        CentralizedCriticModel,
    )

# "lr_schedule": [[0, 0.0005], [20000000, 1e-12]], "num_envs_per_worker": 5, "num_workers": 32, "sample_batch_size": 250, "train_batch_size": 500
    config = {
            'env': env_name,
            # "simple_optimizer": 1,
            'framework': 'tf',

            'batch_mode': 'truncate_episodes',
            'num_workers': sample_cpu-1,
            'num_gpus': sample_gpu,
            'model': {
                "fcnet_hiddens": [256,256,78],
                "fcnet_activation": "relu",
            },
            'multiagent':default_multiagent,

            

            "adam_epsilon": 0.00015, 
            "beta_annealing_fraction": 1.0, 
            "buffer_size": 1000000, 
            "double_q": True, 
            "dueling": True, 
            "env": "BreakoutNoFrameskip-v4", 
            "exploration_final_eps": 0.01, 
            "exploration_fraction": 0.1, 
            "final_prioritized_replay_beta": 1.0, 
            "gpu": True,  
            "lr": 0.0001, 
            "n_step": 3, 
            "noisy": False, 
            "num_atoms": 1, 
            "num_envs_per_worker": 8, 
            "prioritized_replay": False, 
            "prioritized_replay_alpha": 0.5, 
            "sample_batch_size": 158, 
            "schedule_max_timesteps": 2000000, 
            "target_network_update_freq": 50000, 
            "timesteps_per_iteration": 25000, 
            "train_batch_size": 512
        }

    
    if use_callbacks:
        config['callbacks'] = ft.FootballCallbacks

    scheduler = None
    stop = {
        "timesteps_total": n_timesteps,
    }

    filename_stem = os.path.basename(__file__).split(".")[0]
    policy_type = 'search' if use_tune_config else \
        'shared' if n_policies == 1 else 'independent'
    scheduler_type = 'asha' if use_scheduler else 'fifo'
    config_type = 'tune' if use_tune_config else 'fixed'
    experiment_name =f"{filename_stem}_{env_name}_{policy_type}_{n_timesteps}_{scheduler_type}_{config_type}"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    local_dir = os.path.join(script_dir, '..', '..', 'logs')
    a = tune.run(
        'APEX',
        name=experiment_name,
        reuse_actors=False,
        scheduler=scheduler,
        raise_on_failed_trial=True,
        fail_fast=True,
        max_failures=0,
        num_samples=n_samples,
        stop=stop,
        checkpoint_freq=100,
        checkpoint_at_end=True,
        local_dir=local_dir,
        config=config,
        verbose=1 if not debug else 3
    )

    checkpoint_path = a.get_best_checkpoint(a.get_best_trial("episode_reward_mean", "max"), "episode_reward_mean", "max")
    print('Best checkpoint found:', checkpoint_path)
    ray.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "Script for training RLDM's P3 baseline agents",
        formatter_class=RawTextHelpFormatter,
        epilog=EXAMPLE_USAGE)

    parser.add_argument('-c','--num-cpus', default=None, type=int,
                        help='Number of cpus to allocate for experiment.\
                            \nDefault: use all cpus on system.')
    parser.add_argument('-g','--num-gpus', default=None, type=int,
                        help='Number of gpus to allocate for experiment.\
                            \nDefault: use all gpus on system.')
    parser.add_argument('-n','--num-samples', default=1, type=int,
                        help='Number of training samples to run.\
                            \nDefault: 1 sample.')
    parser.add_argument('-o','--at-once', default=False, action='store_true',
                        help='Whether to run all samples at once.\
                            \nDefault: sequential trials.')
    parser.add_argument('-m','--simultaneous-samples', default=None, type=int,
                        help='Number of samples to run simultaneously.\
                            \nDefault: sequential trials.')
    parser.add_argument('-t','--num-timesteps', default=None, type=int,
                        help="Number of environment timesteps to train each sample.\
                            \nDefault: 5_000_000 steps per number of players,\
                            \nand an additional 25%% of the total for independent policies.")
    parser.add_argument('-s','--shared-policy', default=False, action='store_true',
                        help='Whether to train a shared policy for all players.\
                            \nDefault: independent policies.')
    parser.add_argument('-a','--scheduler', default=False, action='store_true',
                        help='Use an ASHA scheduler to run only promising trials.\
                            \nDefault: ASHA scheduler disabled.')
    parser.add_argument('-b','--callbacks', default=False, action='store_true',
                        help='Enable callbacks to display metrics on TensorBoard.\
                            \nDefault: Callbacks disabled.')
    parser.add_argument('-e','--tune', default=False, action='store_true',
                        help='Use tune to search for best hyperparameter combination.\
                            \nDefault: Fixed hyperparameters.')
    parser.add_argument('-r','--dry-run', default=False, action='store_true',
                        help='Print the training plan, and exit.\
                            \nDefault: normal mode.')
    parser.add_argument('-d','--debug', default=False, action='store_true',
                        help='Set full script to debug.\
                            \nDefault: "INFO" output mode.')
    args = parser.parse_args()

    assert args.num_samples <= 100, "Can only train up-to 100 samples on a single run"
    assert args.num_samples >= 1, "Must train at least 1 sample"

    assert not args.at_once or args.num_samples <= 10, "Can only train up-to 10 samples with --at-once"
    assert not args.at_once or args.simultaneous_samples is None, "Use either --at-once or --simultaneous-samples"

    assert not args.tune or not args.shared_policy, "Setting --tune searches for shared policy as well"

    if args.simultaneous_samples:
        assert args.simultaneous_samples <= args.num_samples, "Cannot run more simultaneous than total samples"
        assert args.num_samples > 1, "Must train at least 2 samples if selecting simultaneous"

    assert not args.at_once or args.num_samples <= 4, "Cannot run all samples at once with more than 4 samples"
    assert not args.scheduler or args.num_samples > 2, "Scheduler only makes sense with more than 2 samples"

    n_cpus, n_gpus = st.get_cpu_gpu_count()
    assert args.num_cpus is None or args.num_cpus <= n_cpus, "Didn't find enough cpus"
    assert args.num_gpus is None or args.num_gpus <= n_gpus, "Didn't find enough gpus"
    n_cpus = n_cpus if args.num_cpus is None else args.num_cpus
    n_gpus = n_gpus if args.num_gpus is None else args.num_gpus

    num_players = 3 # hard-coding 3 here for now
    env_name = ft.n_players_to_env_name(num_players, True) # hard-coding auto GK
    n_policies = 1 if args.shared_policy else num_players - 1 # hard-coding

    n_timesteps = args.num_timesteps if args.num_timesteps else num_players * 5_000_000
    if args.num_timesteps is None:
        n_timesteps = int(n_timesteps + n_timesteps * 0.25 * (n_policies > 1))

    sample_cpu = n_cpus if not args.at_once else n_cpus // args.num_samples
    assert sample_cpu >= 1, "Each sample needs at least one cpu to run"
    sample_gpu = 0 if not n_gpus else 1 if not args.at_once or args.num_samples == 1 else n_gpus / args.num_samples

    if args.simultaneous_samples:
        sample_cpu = n_cpus // args.simultaneous_samples
        assert sample_cpu >= 1, "Each sample needs at least one cpu to run"
        sample_gpu = 0 if not n_gpus else n_gpus / args.simultaneous_samples

    print("")
    print("Calling training code with following parameters (arguments affecting each):")
    print("\tEnvironment to load with GFootball:", env_name)
    print("\tNumber of cpus to allocate for RLlib (-c):", n_cpus)
    print("\tNumber of gpus to allocate for RLlib (-g):", n_gpus)
    print("\tNumber of policies to train (-s):", n_policies)
    print("\tNumber of environment timesteps to train (-t):", n_timesteps)
    print("\tNumber of samples to run (-n):", args.num_samples)
    print("\tNumber of simultaneous samples to run (-n, -m):", args.simultaneous_samples)
    print("\tNumber of cpus to allocate for each sample: (-c, -n, -o, -m)", sample_cpu)
    print("\tNumber of gpus to allocate for each sample: (-g, -n, -o, -m)", sample_gpu)
    print("\tActive scheduler (-a):", args.scheduler)
    print("\tActive callbacks (-b):", args.callbacks)
    print("\tTune hyperparameters (-e):", args.tune)
    print("\tIs this a dry-run only (-r):", args.dry_run)
    print("\tScript running on debug mode (-d):", args.debug)
    print("")

    if not args.dry_run:
        main(n_cpus, n_gpus, env_name,
             n_policies, n_timesteps, args.num_samples,
             sample_cpu, sample_gpu, args.scheduler,
             args.tune, args.callbacks, args.debug)
