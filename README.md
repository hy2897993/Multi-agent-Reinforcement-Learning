## Important: Because the whole docker container file is too big, I only uploaded my modified files and created logs. Please replace your "logs","rldm" folders with my folders in the container directory,  then run it.

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
