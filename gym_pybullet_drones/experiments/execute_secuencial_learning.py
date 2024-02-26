#!/usr/bin/env python3
import numpy as np

from gym_pybullet_drones.envs import SimBasicReward
from gym_pybullet_drones.experiments.learning_script import run_learning
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

run_learning(env_name=SimBasicReward,
             num_episodes=int(1e3),
             env_parameters=dict(
                 obs=ObservationType('kin'),
                 act=ActionType('rpm'),
                 initial_xyzs=np.array([[np.random.randint(-10, 10),
                                         np.random.randint(-10, 10),
                                         np.random.randint(0, 10)]]),
                 target_xyzs=np.array([np.random.randint(-10, 10),
                                       np.random.randint(-10, 10),
                                       np.random.randint(0, 10)]),
                 target_rpys=np.array([0, 0, 1.7])
             ),
             )
