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
from rldm.utils import football_tools_modified as ft
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
    # COMMON_CONFIG: TrainerConfigDict = {
    #     "num_workers": 2,
    #     "num_envs_per_worker": 1,
    #     "create_env_on_driver": False,
    #     "rollout_fragment_length": 200,
    #     "batch_mode": "truncate_episodes",
    #     "gamma": 0.99,
    #     "lr": 0.0001,
    #     "train_batch_size": 200,
    #     'model': {
    #         'vf_share_layers': "true",
    #         'use_lstm': "true",
    #         'max_seq_len': 13,
    #         'fcnet_hiddens': [256, 256],
    #         'fcnet_activation': "tanh",
    #         'lstm_cell_size': 256,
    #         'lstm_use_prev_action': "true",
    #         'lstm_use_prev_reward': "true",
    #     },
    #     "optimizer": {},
    #     "horizon": None,
    #     "soft_horizon": False,
    #     "no_done_at_end": False,
    #     "env": None,
    #     "observation_space": None,
    #     "action_space": None,
    #     "env_config": {},
    #     "remote_worker_envs": False,
    #     "remote_env_batch_wait_ms": 0,
    #     "env_task_fn": None,
    #     "render_env": False,
    #     "record_env": False,
    #     "clip_rewards": None,
    #     "normalize_actions": True,
    #     "clip_actions": False,
    #     "preprocessor_pref": "deepmind",
    #     "log_level": "WARN",
    #     # "callbacks": ft.FootballCallbacks,
    #     "ignore_worker_failures": False,
    #     "recreate_failed_workers": False,
    #     "log_sys_usage": True,
    #     "fake_sampler": False,
    #     "framework": "tf",
    #     "eager_tracing": False,
    #     "eager_max_retraces": 20,
    #     "explore": True,
    #     "exploration_config": {
    #         "type": "StochasticSampling",
    #     },
    #     "evaluation_interval": None,
    #     "evaluation_duration": 10,
    #     "evaluation_duration_unit": "episodes",
    #     "evaluation_parallel_to_training": False,
    #     "in_evaluation": False,
    #     "evaluation_config": {
    #     },
    #     "evaluation_num_workers": 0,
    #     "custom_eval_function": None,
    #     "always_attach_evaluation_results": False,
    #     "keep_per_episode_custom_metrics": False,
    #     "sample_async": False,
    #     # "sample_collector": SimpleListCollector,
    #     "observation_filter": "NoFilter",
    #     "synchronize_filters": True,
    #     "tf_session_args": {
    #         "intra_op_parallelism_threads": 2,
    #         "inter_op_parallelism_threads": 2,
    #         "gpu_options": {
    #             "allow_growth": True,
    #         },
    #         "log_device_placement": False,
    #         "device_count": {
    #             "CPU": 1
    #         },
    #         "allow_soft_placement": True,
    #     },
    #     "local_tf_session_args": {
    #         "intra_op_parallelism_threads": 8,
    #         "inter_op_parallelism_threads": 8,
    #     },
    #     "compress_observations": False,
    #     "metrics_episode_collection_timeout_s": 180,
    #     "metrics_num_episodes_for_smoothing": 100,
    #     # "min_time_s_per_reporting": None,
    #     "min_train_timesteps_per_reporting": None,
    #     "min_sample_timesteps_per_reporting": None,
    #     "seed": None,
    #     "extra_python_environs_for_driver": {},
    #     "extra_python_environs_for_worker": {},
    #     "num_gpus": 0,
    #     # "_fake_gpus": False,
    #     # "num_cpus_per_worker": 1,
    #     # "num_gpus_per_worker": 0,
    #     "custom_resources_per_worker": {},
    #     # "num_cpus_for_driver": 1,
    #     "placement_strategy": "PACK",
    #     "input": "sampler",
    #     "input_config": {},
    #     "actions_in_input_normalized": False,
    #     "input_evaluation": ["is", "wis"],
    #     "postprocess_inputs": False,
    #     "shuffle_buffer_size": 0,
    #     "output": None,
    #     "output_config": {},
    #     "output_compress_columns": ["obs", "new_obs"],
    #     "output_max_file_size": 64 * 1024 * 1024,

    #     "multiagent": default_multiagent,
    #     #  {
    #     #     "policies": {},
    #     #     "policy_map_capacity": 100,
    #     #     "policy_map_cache": None,
    #     #     "policy_mapping_fn": None,
    #     #     "policies_to_train": None,
    #     #     "observation_fn": None,
    #     #     "replay_mode": "independent",
    #     #     "count_steps_by": "env_steps",
    #     # },
    #     "logger_config": None,
    #     "_tf_policy_handles_more_than_one_loss": False,
    #     "_disable_preprocessor_api": False,
    #     "_disable_action_flattening": False,
    #     "_disable_execution_plan_api": False,
    #     "disable_env_checking": False,
    #     "simple_optimizer": 0,
    #     # "monitor": DEPRECATED_VALUE,
    #     # "evaluation_num_episodes": DEPRECATED_VALUE,
    #     # "metrics_smoothing_episodes": DEPRECATED_VALUE,
    #     # "timesteps_per_iteration": 0,
    #     # "min_iter_time_s": DEPRECATED_VALUE,
    #     # "collect_metrics_timeout": DEPRECATED_VALUE,
    # }


    
# {"env": "QbertNoFrameskip-v4", "lr_schedule": [[0, 0.0005], [20000000, 1e-12]], "num_envs_per_worker": 5, "num_workers": 32, "sample_batch_size": 250, "train_batch_size": 500}


    config_1 = {
        'env': env_name,
        "vtrace": True,
        "vtrace_clip_rho_threshold": 1.0,
        "vtrace_clip_pg_rho_threshold": 1.0,
        # "vtrace_drop_last_ts":False,
        "num_multi_gpu_tower_stacks": 1,
        "minibatch_buffer_size": 1,
        "replay_proportion": 0.1,
        "replay_buffer_num_slots": 1,
        "learner_queue_size": 16,
        "learner_queue_timeout": 300,

        "grad_clip": 40.0,
        "opt_type": "adam",

        "decay": 0.99,
        "momentum": 0.0,
        "epsilon": 0.1,
        "entropy_coeff_schedule": None,

        "after_train_step": None,

        "simple_optimizer": 1,

        'framework': 'torch',
        'lr': 0.00003,
        'gamma': 0.995,
        # 'lambda': 0.9517171675473532,
        # 'kl_target': 0.010117093480119358,
        # 'kl_coeff': 1.0,
        # 'clip_param': 0.20425701146213993,
        'vf_loss_coeff': 0.3503035138680095,
        # 'vf_clip_param': 1.4862186106326711,
        'entropy_coeff': 0.00087453,
        'num_sgd_iter': 500,
        'train_batch_size': 2_800,
        # "sample_batch_size": 250,
        'rollout_fragment_length': 100,
        # 'sgd_minibatch_size': 128,
        'num_workers': sample_cpu - 1, # one goes to the trainer
        # 'num_envs_per_worker': 5,
        'num_gpus': sample_gpu,
        'batch_mode': 'truncate_episodes',
        'observation_filter': 'NoFilter',
        'log_level': 'INFO' if not debug else 'DEBUG',
        'ignore_worker_failures': False,
        'horizon': 100,
        'model': {
            "fcnet_hiddens": [256,256],
            "fcnet_activation": "relu",
        },
        'multiagent': default_multiagent,
    }
# "lr_schedule": [[0, 0.0005], [20000000, 1e-12]], "num_envs_per_worker": 5, "num_workers": 32, "sample_batch_size": 250, "train_batch_size": 500
    config = {
            'env': env_name,
            "simple_optimizer": 1,
            "vtrace": True,
            "lr_schedule": [[0, 0.0005], [5000000, 1e-12]],
            'batch_mode': 'truncate_episodes',
            'vf_loss_coeff': 0.002,
            'num_workers': sample_cpu-1,
            'num_gpus': sample_gpu,
            "env_config":{
                "config":{
                    "num_agents": 2
                }
            },
            'multiagent':default_multiagent,

            "clip_rewards": True,
            "framework": "torch",
            "model": {
                "fcnet_activation": "linear",
                "fcnet_hiddens": [
                128,
                256,
                64
                ],
                "vf_share_layers": True
            },

            "num_sgd_iter": 1,
            "rollout_fragment_length": 100,
            "train_batch_size": 500,
        }

    if use_tune_config:
        tune_config = {
            'lr': tune.uniform(0.00005, 0.0005),
            'gamma': tune.uniform(0.99, 0.999),
            'lambda': tune.sample_from(
                lambda _: np.clip(np.random.normal(0.95, 0.02), 0.9, 1.0)),
            'kl_target': tune.uniform(0.001, 0.2),
            'kl_coeff': tune.sample_from(
                lambda _: np.clip(np.random.normal(1.00, 0.02), 0.5, 1.0)),
            'clip_param': tune.sample_from(
                lambda _: np.clip(np.random.normal(0.2, 0.01), 0.1, 0.5)),
            'vf_loss_coeff': tune.uniform(0.1, 1.0),
            'vf_clip_param': tune.uniform(0.1, 10.0),
            'entropy_coeff': tune.uniform(0.0, 0.05),
            'num_sgd_iter': tune.randint(5, 20),
            'model': {
                'vf_share_layers': tune.choice(["true", "false"]),
                'use_lstm': tune.choice(["true", "false"]),
                'max_seq_len': tune.qrandint(10, 20),
                'fcnet_hiddens': tune.sample_from(
                    lambda _: random.sample([
                        [256, 256],
                        [128, 256],
                        [256, 128],
                        [128, 128],
                    ], 1)[0]),
                'fcnet_activation': tune.choice(["tanh", "relu"]),
                'lstm_cell_size': tune.choice([128, 256]),
                'lstm_use_prev_action': tune.choice(["true", "false"]),
                'lstm_use_prev_reward': tune.choice(["true", "false"]),
            },
            'multiagent': tune.choice([default_multiagent, shared_multiagent]),
        }
        config = deep_merge(config, config)
    
    if use_callbacks:
        config['callbacks'] = ft.FootballCallbacks

    scheduler = None
    stop = {
        "timesteps_total": n_timesteps,
    }
    if use_scheduler:
        scheduler = ASHAScheduler(
            time_attr='timesteps_total',
            metric='episode_reward_mean',
            mode='max',
            max_t=n_timesteps,
            grace_period=int(n_timesteps*0.10),
            reduction_factor=3,
            brackets=1)
        stop = None

    filename_stem = os.path.basename(__file__).split(".")[0]
    policy_type = 'search' if use_tune_config else \
        'shared' if n_policies == 1 else 'independent'
    scheduler_type = 'asha' if use_scheduler else 'fifo'
    config_type = 'tune' if use_tune_config else 'fixed'
    experiment_name =f"{filename_stem}_{env_name}_{policy_type}_{n_timesteps}_{scheduler_type}_{config_type}"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    local_dir = os.path.join(script_dir, '..', '..', 'logs')
    a = tune.run(
        'IMPALA',
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
# 2022-07-21 00:57:57,046 INFO tune.py:561 -- Total run time: 75520.55 seconds (75517.82 seconds for the tuning loop).
# Best checkpoint found: /mnt/logs/train_agents_IMPALA_3_vs_3_auto_GK_independent_10000000_fifo_fixed/IMPALA_3_vs_3_auto_GK_5308c_00000_0_2022-07-20_03-59-16/checkpoint_001800/checkpoint-1800


# 2022-07-22 00:35:24,549 INFO tune.py:561 -- Total run time: 25255.64 seconds (25254.55 seconds for the tuning loop).
# Best checkpoint found: /mnt/logs/train_agents_IMPALA_3_vs_3_auto_GK_independent_5000000_fifo_fixed/IMPALA_3_vs_3_auto_GK_5f8c8_00000_0_2022-07-21_17-34-29/checkpoint_001900/checkpoint-1900