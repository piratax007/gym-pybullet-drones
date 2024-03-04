#!/usr/bin/env python3

from gym_pybullet_drones.envs import BasicRewardWithPitchRollPenalty
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""######### Learning with BasicRewardWithR Environment #########
R(t) = 25 - 15te - 100Bo - 5R
Target = [0 0 1 0 0 1.7]
Time steps = 10e6
Episode Length = 15s""")

run_learning(env_name=BasicRewardWithR,
             learning_id="Basic-Reward-With-Pitch-Roll-Penalty",
             continuous_learning=True
             )
