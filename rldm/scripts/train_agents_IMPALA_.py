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

    



    config = {

        "_fake_gpus": False,
        "action_space": None,
        "actions_in_input_normalized": False,
        "after_train_step": None,
        "batch_mode": "truncate_episodes",
        "callbacks": "<class 'rldm.utils.football_tools.FootballCallbacks'>",
        "clip_actions": False,
        "clip_rewards": None,
        "collect_metrics_timeout": 180,
        "compress_observations": False,
        "create_env_on_driver": False,
        "custom_eval_function": None,
        "custom_resources_per_worker": {},
        "decay": 0.99,
        "eager_tracing": False,
        "entropy_coeff": 0.0004158966184268587,
        "entropy_coeff_schedule": None,
        "env": "3_vs_3_auto_GK",
        "env_config": {},
        "env_task_fn": None,
        "epsilon": 0.1,
        "evaluation_config": {},
        "evaluation_interval": None,
        "evaluation_num_episodes": 10,
        "evaluation_num_workers": 0,
        "evaluation_parallel_to_training": False,
        "exploration_config": {
            "type": "StochasticSampling"
        },
        "explore": True,
        "extra_python_environs_for_driver": {},
        "extra_python_environs_for_worker": {},
        "fake_sampler": False,
        "framework": "torch",
        "gamma": 0.9936809332376452,
        "grad_clip": 40.0,
        "horizon": 500,
        "ignore_worker_failures": False,
        "in_evaluation": False,
        "input": "sampler",
        "input_config": {},
        "input_evaluation": [
            "is",
            "wis"
        ],
        "learner_queue_size": 16,
        "learner_queue_timeout": 3000,
        "local_tf_session_args": {
            "inter_op_parallelism_threads": 8,
            "intra_op_parallelism_threads": 8
        },
        "log_level": "INFO",
        "log_sys_usage": True,
        "logger_config": None,
        "lr": 0.00022602718266055705,
        "lr_schedule": None,
        "metrics_smoothing_episodes": 100,
        "min_iter_time_s": 0,
        "minibatch_buffer_size": 1,
        "model": {
            "_time_major": False,
            "_use_default_native_models": False,
            "attention_dim": 64,
            "attention_head_dim": 32,
            "attention_init_gru_gate_bias": 2.0,
            "attention_memory_inference": 50,
            "attention_memory_training": 50,
            "attention_num_heads": 1,
            "attention_num_transformer_units": 1,
            "attention_position_wise_mlp_dim": 32,
            "attention_use_n_prev_actions": 0,
            "attention_use_n_prev_rewards": 0,
            "conv_activation": "relu",
            "conv_filters": None,
            "custom_action_dist": None,
            "custom_model": None,
            "custom_model_config": {},
            "custom_preprocessor": None,
            "dim": 84,
            "fcnet_activation": "tanh",
            "fcnet_hiddens": [
            256,
            256
            ],
            "framestack": True,
            "free_log_std": False,
            "grayscale": False,
            "lstm_cell_size": 256,
            "lstm_use_prev_action": "True",
            "lstm_use_prev_action_reward": -1,
            "lstm_use_prev_reward": "True",
            "max_seq_len": 13,
            "no_final_linear": False,
            "num_framestacks": "auto",
            "post_fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "use_attention": False,
            "use_lstm": "True",
            "vf_share_layers": "True",
            "zero_mean": True
        },
        "momentum": 0.0,
        "monitor": -1,
        "multiagent": {
            "count_steps_by": "env_steps",
            "observation_fn": None,
            "policies": {
            "agent_0": [
                None,
                "Box([-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf\n -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf\n -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf\n -inf], [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf\n inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf\n inf inf inf inf inf inf inf], (43,), float32)",
                "Discrete(19)",
                {}
            ],
            "agent_1": [
                None,
                "Box([-inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf\n -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf\n -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf -inf\n -inf], [inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf\n inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf inf\n inf inf inf inf inf inf inf], (43,), float32)",
                "Discrete(19)",
                {}
            ]
            },
            "policies_to_train": None,
            "policy_map_cache": None,
            "policy_map_capacity": 100,
            "policy_mapping_fn": "<function main.<locals>.<lambda> at 0x7f28deedce60>",
            "replay_mode": "independent"
        },
        "no_done_at_end": False,
        "normalize_actions": True,
        "num_cpus_for_driver": 1,
        "num_cpus_per_worker": 1,
        "num_envs_per_worker": 5,
        "num_gpus": 1,
        "num_gpus_per_worker": 0,
        "num_sgd_iter": 16,
        "num_workers": 7,
        "observation_filter": "NoFilter",
        "observation_space": None,
        "opt_type": "adam",
        "optimizer": {},
        "output": None,
        "output_compress_columns": [
            "obs",
            "new_obs"
        ],
        "output_max_file_size": 67108864,
        "placement_strategy": "PACK",
        "postprocess_inputs": False,
        "preprocessor_pref": "deepmind",
        "record_env": False,
        "remote_env_batch_wait_ms": 0,
        "remote_worker_envs": False,
        "render_env": False,
        "replay_buffer_num_slots": 0,
        "replay_proportion": 0.0,
        "rollout_fragment_length": 100,
        "sample_async": False,
        "sample_collector": "<class 'ray.rllib.evaluation.collectors.simple_list_collector.SimpleListCollector'>",
        "seed": None,
        "shuffle_buffer_size": 0,
        "simple_optimizer": 1,
        "soft_horizon": False,
        "synchronize_filters": True,
        "tf_session_args": {
            "allow_soft_placement": True,
            "device_count": {
            "CPU": 1
            },
            "gpu_options": {
            "allow_growth": True
            },
            "inter_op_parallelism_threads": 2,
            "intra_op_parallelism_threads": 2,
            "log_device_placement": False
        },
        "timesteps_per_iteration": 0,
        "train_batch_size": 500,
        "vf_loss_coeff": 0.3503035138680095,
        "vtrace": True,
        "vtrace_clip_pg_rho_threshold": 1.0,
        "vtrace_clip_rho_threshold": 1.0
        
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
