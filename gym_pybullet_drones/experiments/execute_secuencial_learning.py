#!/usr/bin/env python3

from gym_pybullet_drones.envs import SimBasicReward
from gym_pybullet_drones.experiments.learning_script import run_learning

print("######### Learning with SimBasicReward Environment #########")
# R(t) = 25 - 15te - 100Bo
# Target = [0 0 1 0 0 1.7]
run_learning(env_name=SimBasicReward,
             learning_id="SimBasicReward",
             num_episodes=int(5e6),
             )
