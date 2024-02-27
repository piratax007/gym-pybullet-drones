#!/usr/bin/env python3

from gym_pybullet_drones.envs import SimBasicReward
from gym_pybullet_drones.experiments.learning_script import run_learning

run_learning(env_name=SimBasicReward,
             num_episodes=int(1e3),
             )
