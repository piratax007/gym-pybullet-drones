#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
from stable_baselines3 import PPO

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.envs import FromScratch


def in_degrees(angles):
    return list(map(lambda angle: angle * 180 / np.pi, angles))


def get_policy(policy_path):
    if os.path.isfile(policy_path + '/best_model.zip'):
        return PPO.load(policy_path + '/best_model.zip')

    raise Exception("[ERROR]: no model under the specified path", policy_path)


def run_simulation(policy_path, test_env, gui=True, record_video=False, save=False, comment=""):
    policy = get_policy(policy_path)

    test_env = test_env(gui=gui,
                        obs=ObservationType('kin'),
                        act=ActionType('rpm'),
                        record=record_video)

    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1,
        output_folder=policy_path,
        colab=False
    )

    obs, info = test_env.reset(seed=42, options={})
    log_reward = []
    simulation_length = (test_env.EPISODE_LEN_SEC + 92) * test_env.CTRL_FREQ

    start = time.time()

    for i in range(simulation_length):
        # if i < (simulation_length / 5):
        #     z_target = 1
        # elif i < 2 * (simulation_length / 5):
        #     z_target = 0.2
        # elif i < 3 * (simulation_length / 5):
        #     z_target = 0.8
        # elif i < 4 * (simulation_length / 5):
        #     z_target = 0.4
        # else:
        #     z_target = 1
        #
        # obs[0][2] += 1 - z_target
        action, _states = policy.predict(obs,
                                         deterministic=True
                                         )
        obs, reward, terminated, truncated, info = test_env.step(action)
        log_reward.append(reward)
        actions = test_env._getDroneStateVector(0)[16:20]
        actions2 = actions.squeeze()
        obs2 = obs.squeeze()
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

        logger.log(
            drone=0,
            timestamp=i / test_env.CTRL_FREQ,
            state=np.hstack([obs2[0:3],
                             np.zeros(4),
                             obs2[3:12],
                             actions2
                             ]),
            control=np.zeros(12)
        )

        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)

    test_env.close()

    if save:
        logger.save_as_csv(comment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a simulation given a trained policy")
    parser.add_argument(
        '--policy_path',
        help='The path to a zip file containing the trained policy'
    )
    parser.add_argument(
        '--test_env',
        default=FromScratch,
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
    parser.add_argument(
        '--save',
        default=False,
        type=str2bool,
        help='Allow to save the trained data using csv and npy files'
    )
    parser.add_argument(
        '--comment',
        default="",
        type=str,
        help="A comment to describe de simulation saved data"
    )

    run_simulation(**vars(parser.parse_args()))
