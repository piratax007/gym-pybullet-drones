#!/usr/bin/env python3

from gym_pybullet_drones.envs import SimBasicReward, RewardRPMsAndC9
from gym_pybullet_drones.experiments.learning_script import run_learning

print("######### Learning with RewardRPMsAndC9 Environment #########")
run_learning(env_name=RewardRPMsAndC9,
             num_episodes=int(1e6),
             )

print("######### Learning with SimBasicReward Environment #########")
run_learning(env_name=SimBasicReward,
             num_episodes=int(1e6),
             )
