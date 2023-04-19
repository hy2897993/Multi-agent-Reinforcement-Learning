from __future__ import absolute_import, division, print_function
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import argparse
import os

import warnings ; warnings.filterwarnings('ignore')
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import collections
import os
import random
import tempfile
from argparse import RawTextHelpFormatter

import gfootball.env as football_env
import gym
import numpy as np
import ray
import ray.cloudpickle as cloudpickle
import torch
from gfootball import env as fe
from gym import wrappers
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.tune.registry import get_trainable_cls, register_env
from ray.tune.schedulers import ASHAScheduler
from rldm.utils import football_tools as ft
from rldm.utils import gif_tools as gt
from rldm.utils import system_tools as st

EXAMPLE_USAGE = """
Example usage:

    Test a checkpoint for 10 episodes on the original scenario:
        python -m rldm.scripts.evaluate_checkpoint -c <path to checkpoint>
        NOTE: Checkpoints will look something like the following:
            "/mnt/logs/baselines/baseline_1/checkpoint_0/checkpoint-0"

    Test a checkpoint for 10 episodes:
        python -m rldm.scripts.evaluate_checkpoint -c <path to checkpoint>

    Test a checkpoint for 100 episodes:
        python -m rldm.scripts.evaluate_checkpoint -c <path to checkpoint> -e 100

    Test multiple checkpoints for 10 episodes and create a graph:
        python -m rldm.scripts.evaluate_checkpoint -c <path to json> -g
    
    Create a barplot 

    NOTE:
        There are a few baseline checkpoints available for you to test:
          -> /mnt/logs/baselines/baseline_1/checkpoint_0/checkpoint-0
        To test all the baseline checkpoints at once use the json file provided:
          -> /mnt/rldm/scripts/checkpoints.json
          Data is formatted as { "<Name of checkpoint>": ["<algorithm or model to load>", "<path to checkpoint>"]}

        Test them:
          python -m rldm.scripts.evaluate_checkpoint -c /mnt/logs/baselines/baseline_1/checkpoint_0/checkpoint-0
"""


class DefaultMapping(collections.defaultdict):
    """default_factory now takes as an argument the missing key."""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value

STAT_FUNC = {
    'min': np.min,
    'mean': np.mean,
    'median': np.median,
    'max': np.max,
    'std': np.std,
    'var': np.var,
}

def main(checkpoint, algorithm, env_name, config, num_episodes, debug):

    ray.init(log_to_driver=debug, include_dashboard=False,
             local_mode=True, logging_level='DEBUG' if debug else 'ERROR')
    register_env(env_name, lambda _: ft.RllibGFootball(env_name=env_name))

    cls = get_trainable_cls(algorithm)
    agent = cls(env=env_name, config=config)
    agent.restore(checkpoint)

    env = ft.RllibGFootball(env_name=env_name)

    policy_agent_mapping = agent.config["multiagent"]["policy_mapping_fn"]
    policy_map = agent.workers.local_worker().policy_map
    state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
    use_lstm = {p: len(s) > 0 for p, s in state_init.items()}

    action_init = {
        p: flatten_to_single_ndarray(m.action_space.sample())
        for p, m in policy_map.items()
    }

    eps_stats = {}
    eps_stats['rewards_total'] = []
    eps_stats['timesteps'] = []
    eps_stats['score_reward'] = []
    eps_stats['win_perc'] = []
    eps_stats['undefeated_perc'] = []
    for eidx in range(num_episodes):
        mapping_cache = {}  # in case policy_agent_mapping is stochastic
        
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda player_id: state_init[mapping_cache[player_id]])
        prev_actions = DefaultMapping(
            lambda player_id: action_init[mapping_cache[player_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)

        done = False
        eps_stats['rewards_total'].append({a:0.0 for a in obs})
        eps_stats['timesteps'].append(-1)
        eps_stats['score_reward'].append(0)
        eps_stats['win_perc'].append(0)
        eps_stats['undefeated_perc'].append(0)
        while not done:
            actions = {}
            for player_id, a_obs in obs.items():
                policy_id = mapping_cache.setdefault(
                    player_id, policy_agent_mapping(player_id, None))
                p_use_lstm = use_lstm[policy_id]
                if p_use_lstm:
                    a_action, p_state, _ = agent.compute_single_action(
                        a_obs,
                        state=agent_states[player_id],
                        prev_action=prev_actions[player_id],
                        prev_reward=prev_rewards[player_id],
                        policy_id=policy_id)
                    agent_states[player_id] = p_state
                else:
                    a_action = agent.compute_single_action(
                        a_obs,
                        prev_action=prev_actions[player_id],
                        prev_reward=prev_rewards[player_id],
                        policy_id=policy_id)

                a_action = flatten_to_single_ndarray(a_action)
                actions[player_id] = a_action
                prev_actions[player_id] = a_action

            next_obs, rewards, dones, infos = env.step(actions)

            done = dones['__all__']
            for player_id, r in rewards.items():
                prev_rewards[player_id] = r
                eps_stats['rewards_total'][-1][player_id] += r

            obs = next_obs
            eps_stats['timesteps'][-1] += 1

        eps_stats['score_reward'][-1] = infos['player_0']['score_reward']
        game_result = "loss" if infos['player_0']['score_reward'] == -1 else \
            "win" if infos['player_0']['score_reward'] == 1 else "tie"
        eps_stats['win_perc'][-1] = int(game_result == "win")
        eps_stats['undefeated_perc'][-1] = int(game_result != "loss")
        print(f"\nEpisode #{eidx+1} ended with a {game_result}:")
        for p, r in eps_stats['rewards_total'][-1].items():
            print("\t{} got episode reward: {:.2f}".format(p, r))
        print("\tTotal reward: {:.2f}".format(sum(eps_stats['rewards_total'][-1].values())))

    eps_stats['rewards_total'] = {k: [dic[k] for dic in eps_stats['rewards_total']] \
        for k in eps_stats['rewards_total'][0]}
    print("\n\nAll trials completed:")
    for stat_name, values in eps_stats.items():
        print(f"\t{stat_name}:")
        if type(values) is dict:
            for stat_name2, values2 in values.items():
                print(f"\t\t{stat_name2}:")
                for func_name, func in STAT_FUNC.items():
                    print("\t\t\t{}: {:.2f}".format(func_name, func(values2)))
        else:
            for func_name, func in STAT_FUNC.items():
                print("\t\t{}: {:.2f}".format(func_name, func(values)))

    ray.shutdown()
    return eps_stats

def createBarplot(data):
    df = pd.DataFrame(data)
    sns.barplot(x='Checkpoint', y='Win Rate', capsize=0.2, data=df).set(title="Win Rate % for each Checkpoint")
    plt.xticks(rotation=40)
    plt.tight_layout()
    plt.savefig('winrate.png')
    plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = "Script for testing RLDM's P3 baseline agents",
        formatter_class=RawTextHelpFormatter,
        epilog=EXAMPLE_USAGE)
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help='[REQUIRED] Checkpoint or json file containing paths to checkpoints from which to roll out.')
    parser.add_argument('-g', '--graph', default=False, action='store_true',
                        help='Create a barplot graph for win perc.')
    parser.add_argument('-a', '--algorithm', type=str, default='PPO',
                        help="The algorithm or model to load. This may refer to the name\
                            \nof a built-on algorithm (e.g. RLLib's DQN or PPO), or a\
                            \nuser-defined trainable function or class registered in the \
                            \ntune registry. If using a json with multiple checkpoints\
                            \nit will be specified there instead.\
                            \nDefault: PPO.")
    parser.add_argument('-e','--num-episodes', default=10, type=int,
                        help='Number of episodes to test your agent(s).\
                            \nDefault: 10 episodes.')
    parser.add_argument('-r','--dry-run', default=False, action='store_true',
                        help='Print the training plan, and exit.\
                            \nDefault: normal mode.')
    parser.add_argument('-d','--debug', default=False, action='store_true',
                        help='Set full script to debug.\
                            \nDefault: "INFO" output mode.')
    args = parser.parse_args()

    assert args.checkpoint, "At least one checkpoint is required"

    assert args.num_episodes <= 100, "Must rollout a maximum of 100 episodes"
    assert args.num_episodes >= 1, "Must rollout at least 1 episode"
    
    n_cpus, _ = st.get_cpu_gpu_count()
    assert n_cpus >= 2, "Didn't find enough cpus"

    if args.checkpoint.endswith(".json"):
        # Loading multiple checkpoints
        import json
        with open(args.checkpoint, "r") as checkpoint_data:
            checkpoints = json.load(checkpoint_data)
    else:
        # Load one checkpoint
        checkpoints = {"Checkpoint": args.checkpoint}
    
    # Accumulate data for each checkpoint
    data = {'Checkpoint':[], 'Win Rate':[]}
    for name, check in checkpoints.items():
        args.algorithm = check[0]
        args.checkpoint = check[1]
        config = {}
        config_dir = os.path.dirname(args.checkpoint)
        config_path = os.path.join(config_dir, "../params.pkl")
        assert os.path.exists(config_path), "Could not find a proper config from checkpoint"
        with open(config_path, "rb") as f:
            config = cloudpickle.load(f)

        config["create_env_on_driver"] = True
        config["log_level"] = 'DEBUG' if args.debug else 'ERROR' 
        config["num_workers"] = 1
        config["num_gpus"] = 0
        test_env_name="3_vs_3_auto_GK" # this semester we're only doing 3v3 w auto GK

        print("")
        print("Calling training code with following parameters (arguments affecting each):")
        print("\tLoading checkpoint (-c):", args.checkpoint)
        print("\tLoading driver (-a):", args.algorithm)
        print("\tEnvironment to load for testing (-l):", test_env_name)
        print("\tNumber of episodes to run (-e):", args.num_episodes)
        print("\tIs this a dry-run only (-r):", args.dry_run)
        print("\tScript running on debug mode (-d):", args.debug)
        print("")

        if not args.dry_run:
            output = main(args.checkpoint, args.algorithm, test_env_name, config, args.num_episodes, args.debug)
            # Gather data required for graph
            if args.graph:
                # X Axis
                data_len = len(output['win_perc'])
                data['Checkpoint'].extend([name for _ in range(data_len)])
                # Y Axis
                data['Win Rate'].extend(output['win_perc'])
    createBarplot(data)
