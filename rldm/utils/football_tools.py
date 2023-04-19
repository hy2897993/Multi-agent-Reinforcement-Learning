import os
from typing import TYPE_CHECKING, Dict, Optional

import gfootball
import gym
import numpy as np
from gfootball import env as fe
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import AgentID, PolicyID


def get_obs_act_space(env_name):
    env = RllibGFootball(env_name=env_name)
    obs_space = env.observation_space
    act_space = env.action_space
    del env
    return obs_space, act_space

def env_name_to_n_players(env_name):
    n_players = int(env_name[0])
    if 'auto_GK' in env_name:
        n_players -= 1
    return n_players

def n_players_to_env_name(n_players, auto_GK):
    env_name = f"{n_players}_vs_{n_players}"
    GK_addon = "_auto_GK" if auto_GK else ""
    env_name += GK_addon
    return env_name

def create_football_env(env_name, n_controls, write_video, render, logdir):
    gfootball_dir = os.path.dirname(gfootball.__file__)
    assert os.path.exists(gfootball_dir), "Couldn't find gfootball package, make sure it is installed"
    scenarios_dir = os.path.join(gfootball_dir, "scenarios")
    assert os.path.exists(scenarios_dir), "Couldn't find gfootball scenarios folder, make sure it is installed"

    scenario_file_name = f"{env_name}.py"
    scenarios_gfootbal_file = os.path.join(scenarios_dir, scenario_file_name)
    if not os.path.exists(scenarios_gfootbal_file):
        local_dir = os.path.dirname(__file__)
        local_scenario_file = os.path.join(local_dir, '..', '..', 'docker', scenario_file_name)
        assert os.path.exists(local_scenario_file), f"Couldn't find {local_scenario_file}, can't copy it to {scenarios_dir}"
        from shutil import copyfile
        copyfile(local_scenario_file, scenarios_gfootbal_file)

    assert os.path.exists(scenarios_gfootbal_file), f"Couldn't find {scenarios_gfootbal_file}, make sure you manually copy {scenario_file_name} to {scenarios_dir}"

    env = fe.create_environment(
        env_name=env_name,
        stacked=False,
        representation='simple115v2',
        # scoring is 1 for scoring a goal, -1 the opponent scoring a goal
        # checkpoint is +0.1 first time player gets to an area (10 checkpoint total, +1 reward max)
        rewards='checkpoints,scoring',
        logdir=logdir,
        write_goal_dumps=write_video,
        write_full_episode_dumps=write_video,
        render=render,
        write_video=write_video,
        dump_frequency=1 if write_video else 0,
        extra_players=None,
        number_of_left_players_agent_controls=n_controls,
        number_of_right_players_agent_controls=0)

    return env


class RllibGFootball(MultiAgentEnv):
    EXTRA_OBS_IDXS = np.r_[6:22,28:44,50:66,72:88,100:108]

    def __init__(self, env_name, write_video=False, render=False, logdir='/tmp/football'):
        self.n_players = env_name_to_n_players(env_name)
        self.env = create_football_env(env_name, self.n_players, write_video, render, logdir)

        self.action_space, self.observation_space = {}, {}
        for idx in range(self.n_players):
            self.action_space[f'player_{idx}'] = gym.spaces.Discrete(self.env.action_space.nvec[idx]) \
                if self.n_players > 1 else self.env.action_space
            lows = np.delete(self.env.observation_space.low[idx], RllibGFootball.EXTRA_OBS_IDXS)
            highs = np.delete(self.env.observation_space.high[idx], RllibGFootball.EXTRA_OBS_IDXS)
            self.observation_space[f'player_{idx}'] = gym.spaces.Box(
                low=lows, high=highs, dtype=self.env.observation_space.dtype) \
                if self.n_players > 1 else self.env.observation_space

        self.reward_range = np.array((-np.inf, np.inf))
        self.metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
        self.spec = None

    def _tidy_obs(self, obs):
        for key, values in obs.items():
            obs[key] = np.delete(values, RllibGFootball.EXTRA_OBS_IDXS)
        return obs

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for idx in range(self.n_players):
            obs[f'player_{idx}'] = original_obs[idx] \
                if self.n_players > 1 else original_obs
        return self._tidy_obs(obs)

    def step(self, action_dict):

        actions = []
        for idx in range(self.n_players):
            actions.append(action_dict[f'player_{idx}'])
        o, r, d, i = self.env.step(actions)

        game_info = {}
        for k, v in self.env.unwrapped._env._observation.items():
            game_info[k] = v

        scenario = self.env.unwrapped._env._config['level']
        obs, rewards, dones, infos = {}, {}, {}, {}
        for idx in range(self.n_players):
            obs[f'player_{idx}'] = o[idx] \
                if self.n_players > 1 else o
            rewards[f'player_{idx}'] = r[idx] \
                if self.n_players > 1 else r
            dones[f'player_{idx}'] = d
            dones['__all__'] = d
            infos[f'player_{idx}'] = i
            infos[f'player_{idx}']['game_scenario'] = scenario
            infos[f'player_{idx}']['game_info'] = game_info
            infos[f'player_{idx}']['action'] = action_dict[f'player_{idx}']

        return self._tidy_obs(obs), rewards, dones, infos


class FootballCallbacks(DefaultCallbacks):
    STAT_FUNC = {
        'min': np.min,
        'mean': np.mean,
        'median': np.median,
        'max': np.max,
        'std': np.std,
        'var': np.var,
    }

    def on_episode_start(self,
                         *,
                         worker: "RolloutWorker",
                         base_env: BaseEnv,
                         policies: Dict[PolicyID, Policy],
                         episode: MultiAgentEpisode,
                         env_index: Optional[int] = None,
                         **kwargs) -> None:

        # you can do things on episode start, if you need to
        pass

    def on_episode_step(self,
                        *,
                        worker: "RolloutWorker",
                        base_env: BaseEnv,
                        episode: MultiAgentEpisode,
                        env_index: Optional[int] = None,
                        **kwargs) -> None:

        last_info = None
        for agent_id in episode.get_agents():
            p_idx = int(agent_id.split("_")[-1])
            obs = episode.last_observation_for(agent_id)
            for obs_idx, obs_value in enumerate(obs):
                key = f"team/player_{p_idx}/observation_{obs_idx}"
                val = obs_value
                episode.user_data.setdefault(key, []).append(val)

            info = episode.last_info_for(agent_id)

            key = f"team/player_{p_idx}/action_{info['action']}_selected"
            val = 1
            episode.user_data.setdefault(key, []).append(val)

            info = info['game_info']

            key = f"team/player_{p_idx}/position_0"
            val = info['left_team'][p_idx][0]
            episode.user_data.setdefault(key, []).append(val)
            key = f"team/player_{p_idx}/position_1"
            val = info['left_team'][p_idx][1]
            episode.user_data.setdefault(key, []).append(val)

            key = f"team/player_{p_idx}/direction_0"
            val = info['left_team_direction'][p_idx][0]
            episode.user_data.setdefault(key, []).append(val)
            key = f"team/player_{p_idx}/direction_1"
            val = info['left_team_direction'][p_idx][1]
            episode.user_data.setdefault(key, []).append(val)

            key = f"team/player_{p_idx}/tired_factor"
            val = info['left_team_tired_factor'][p_idx]
            episode.user_data.setdefault(key, []).append(val)

            key = f"team/player_{p_idx}/active"
            val = info['left_team_active'][p_idx]
            episode.user_data.setdefault(key, []).append(val)

            key = f"team/player_{p_idx}/yellow_card"
            val = info['left_team_yellow_card'][p_idx]
            episode.user_data.setdefault(key, []).append(val)

            key = f"team/player_{p_idx}/roles"
            val = info['left_team_roles'][p_idx]
            episode.user_data.setdefault(key, []).append(val)

            key = f"team/player_{p_idx}/controllable"
            val = info['left_agent_controlled_player'][p_idx]
            episode.user_data.setdefault(key, []).append(val)

            for aid, val in enumerate(info['left_agent_sticky_actions'][p_idx]):
                key = f"team/player_{p_idx}/left_agent_sticky_actions{aid}"
                episode.user_data.setdefault(key, []).append(val)

            last_info = info

        episode.user_data.setdefault("ball_0", []).append(last_info['ball'][0])
        episode.user_data.setdefault("ball_1", []).append(last_info['ball'][1])
        episode.user_data.setdefault("ball_2", []).append(last_info['ball'][2])

        episode.user_data.setdefault("ball_direction_0", []).append(last_info['ball_direction'][0])
        episode.user_data.setdefault("ball_direction_1", []).append(last_info['ball_direction'][1])
        episode.user_data.setdefault("ball_direction_2", []).append(last_info['ball_direction'][2])

        team = 'left' if last_info['ball_owned_team'] == -1 else 'right'
        episode.user_data.setdefault(f"ball_owned_team_{team}", []).append(1)
        player = str(last_info['ball_owned_player'])
        episode.user_data.setdefault(f"ball_owned_team_{team}_player_{player}", []).append(1)

        designated_player = last_info['left_team_designated_player']
        episode.user_data.setdefault(f"team/designated_player_{designated_player}", []).append(1)

        designated_player = last_info['right_team_designated_player']
        episode.user_data.setdefault(f"opponents/designated_player_{designated_player}", []).append(1)

        for p_idx in range(len(last_info['right_team'])):

            key = f"opponents/player_{p_idx}/position_0"
            val = last_info['right_team'][p_idx][0]
            episode.user_data.setdefault(key, []).append(val)
            key = f"opponents/player_{p_idx}/position_1"
            val = last_info['right_team'][p_idx][1]
            episode.user_data.setdefault(key, []).append(val)

            key = f"opponents/player_{p_idx}/direction_0"
            val = last_info['right_team_direction'][p_idx][0]
            episode.user_data.setdefault(key, []).append(val)
            key = f"opponents/player_{p_idx}/direction_1"
            val = last_info['right_team_direction'][p_idx][1]
            episode.user_data.setdefault(key, []).append(val)

            key = f"opponents/player_{p_idx}/tired_factor"
            val = last_info['right_team_tired_factor'][p_idx]
            episode.user_data.setdefault(key, []).append(val)
            key = f"opponents/player_{p_idx}/active"
            val = last_info['right_team_active'][p_idx]
            episode.user_data.setdefault(key, []).append(val)
            key = f"opponents/player_{p_idx}/yellow_card"
            val = last_info['right_team_yellow_card'][p_idx]
            episode.user_data.setdefault(key, []).append(val)
            key = f"opponents/player_{p_idx}/roles"
            val = last_info['right_team_roles'][p_idx]
            episode.user_data.setdefault(key, []).append(val)


    def on_episode_end(self,
                       *,
                       worker: "RolloutWorker",
                       base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode,
                       env_index: Optional[int] = None,
                       **kwargs) -> None:

        last_info = episode.last_info_for(episode.get_agents()[0])
        score_reward = last_info['score_reward']
        game_result = "loss" if last_info['score_reward'] == -1 else \
            "win" if last_info['score_reward'] == 1 else "tie"

        episode.custom_metrics["game_result/score_reward_episode"] = score_reward
        episode.custom_metrics["game_result/win_percentage_episode"] = int(game_result == "win")
        episode.custom_metrics["game_result/undefeated_percentage_episode"] = int(game_result != "loss")

        for key, values in episode.user_data.items():
            for fname, f in FootballCallbacks.STAT_FUNC.items():
                episode.custom_metrics[f"{key}_timestep_{fname}_episode"] = f(values)
