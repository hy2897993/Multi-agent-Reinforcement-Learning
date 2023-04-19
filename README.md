# Multi-agent Reinforcement Learning in Football Environment

Reinforcement learning with a multiagent system is a more complex problem than single-agent reinforcement learning. In multi-agent RL we need to achieve not only maximize the reward but studying how multiple agents interact in the same environment. The interaction between agents can be cooperation, competition, or mixed, depending on the environment and training goal. In this report, I experimented with several multi-agent RL algorithms, including Proximal Policy Optimization (PPO) and Importance-weighted Actor-Learner Architecture (IMPALA), in the google research Football environment. My focus is on maximizing the reward while improving the cooperation behavior of players. To compare these algorithms, I analyzed the training results of the trained agents, and the learning metrics and agents’ behavior statics during training.

# Code Usage

## The Algorithms I implemented are in rldm/scripts folder, to run the algorithms

Run the PPO_CC algorithm:
"python -m rldm.scripts.train_agents_PPO_CC -b -t 5000000"

Run the IMPALA algorithm:
"python -m rldm.scripts.train_agents_IMPALA -b -t 5000000"

## I customized the callback functions to get the metrics plots, the callback file is rldm\utils\football_tools_modified.py

In "train_agents_PPO_CC.py" and "train_agents_IMPALA.py" files, 
I replaced "from rldm.utils import football_tools as ft" 
by "from rldm.utils import football_tools_modified as ft"

## To get the training result plots, 
run:
"python -m rldm.scripts.plotting_training_result"

Then the plots can be found in folder "logs\plots"

## To get the comparison bar chart, use the customized file rldm\scripts\evaluate_checkpoint_modified.py

Run:
python -m rldm.scripts.evaluate_checkpoint_modified -c rldm/scripts/checkpoints.json -e 100 -g

Then the plots can be found in folder "logs\plots"
