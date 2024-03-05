#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
from stable_baselines3 import PPO
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.envs.EnvironmentTest import EnvironmentTest

DEFAULT_TEST_ENV = EnvironmentTest
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_RECORD_VIDEO = False
DEFAULT_OBS = ObservationType('kin')  # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm')


def in_degrees(angles):
    return list(map(lambda angle: angle * 180 / np.pi, angles))


def run_simulation(policy_path, test_env, plot, gui=True, record_video=False):
    model = None
    if os.path.isfile(policy_path+'/best_model.zip'):
        model = PPO.load(policy_path+'/best_model.zip')
    else:
        print("[ERROR]: no model under the specified path", policy_path)

    test_env = test_env(gui=gui,
                        obs=DEFAULT_OBS,
                        act=DEFAULT_ACT,
                        record=record_video)

    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1,
        output_folder=policy_path,
        colab=False
    )

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()

    for i in range((test_env.EPISODE_LEN_SEC+12)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, _, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print(f"""
        #################################################################
        Observation Space:
        Position: {obs[0][0:3]}
        Orientation: {in_degrees(obs[0][3:6])}
        Linear Velocity: {obs[0][6:9]}
        Angular Velocity: {obs[0][9:12]}
        -----------------------------------------------------------------
        Action Space: {action}
        Terminated: {terminated}
        Truncated: {truncated}
        #################################################################
        """)

        if DEFAULT_OBS == ObservationType.KIN:
            logger.log(
                drone=0,
                timestamp=i/test_env.CTRL_FREQ,
                state=np.hstack([obs2[0:3],
                                np.zeros(4),
                                obs2[3:15],
                                act2
                                 ]),
                control=np.zeros(12)
            )

        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)

    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot_position_and_orientation()
        logger.plot_rpms()
        logger.plot_trajectory()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a simulation given a trained policy")
    parser.add_argument(
        '--policy_path',
        help='The path to a zip file containing the trained policy'
    )
    parser.add_argument(
        '--test_env',
        default=DEFAULT_TEST_ENV,
        help='The name of the environment to learn, registered with gym_pybullet_drones'
    )
    parser.add_argument(
        '--plot',
        default=True,
        type=str2bool,
        help='The name of the environment to learn, registered with gym_pybullet_drones'
    )
    parser.add_argument(
        '--gui',
        default=True,
        type=str2bool,
        help='The name of the environment to learn, registered with gym_pybullet_drones'
    )
    parser.add_argument(
        '--record_video',
        default=False,
        type=str2bool,
        help='The name of the environment to learn, registered with gym_pybullet_drones'
    )

    run_simulation(**vars(parser.parse_args()))
