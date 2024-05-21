#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
from stable_baselines3 import PPO
from gym_pybullet_drones.envs import ObS12Stage3
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync, str2bool, FIRFilter


def in_degrees(angles):
    return list(map(lambda angle: angle * 180 / np.pi, angles))


def get_policy(policy_path):
    if os.path.isfile(policy_path + '/best_model.zip'):
        return PPO.load(policy_path + '/best_model.zip')

    raise Exception("[ERROR]: no model under the specified path", policy_path)


def circular_trajectory(number_of_points: int = 50) -> tuple:
    angles = np.linspace(0, 4 * np.pi, number_of_points)
    x_coordinates = 2 * np.cos(angles)
    y_coordinates = 2 * np.sin(angles)
    z_coordinates = np.linspace(0, 1, number_of_points)
    return x_coordinates, y_coordinates, z_coordinates


def target_angles(number_of_points: int = 50) -> tuple:
    return tuple(np.linspace(0, 1.3, number_of_points))


def run_simulation(
        policy_path,
        test_env, gui=True,
        record_video=False,
        reset=False,
        save=False,
        plot=False,
        debug=False,
        apply_filter=False,
        comment=""
):
    policy = get_policy(policy_path)

    test_env = test_env(gui=gui,
                        obs=ObservationType('kin'),
                        act=ActionType('rpm'),
                        initial_xyzs=np.array([[0, 0, 0]]),
                        initial_rpys=np.array([[0, 0, 0]]),
                        record=record_video)

    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1,
        output_folder=policy_path,
        colab=False
    )

    obs, info = test_env.reset(seed=42, options={})
    log_reward = []
    simulation_length = (test_env.EPISODE_LEN_SEC + 62) * test_env.CTRL_FREQ

    start = time.time()

    firfilter = FIRFilter()

    for _ in range(firfilter.buffer_size):
        firfilter.buffer.append(np.zeros((1, 4)))

    for i in range(simulation_length):
        x_target, y_target, z_target = circular_trajectory(simulation_length)

        obs[0][0] += x_target[i]
        obs[0][1] += y_target[i]
        obs[0][2] += (0.25 - z_target[i])
        obs[0][5] += target_angles(simulation_length)[i]

        action, _states = policy.predict(obs,
                                         deterministic=True
                                         )

        if apply_filter:
            action = firfilter.filter_actions(action)

        obs, reward, terminated, truncated, info = test_env.step(action)
        log_reward.append(reward)
        actions = test_env._getDroneStateVector(0)[16:20]
        actions2 = actions.squeeze()
        obs2 = obs.squeeze()

        if debug:
            print(f"""
            #################################################################
            Observation Space:
            Position: {obs[0][0:3]}
            Orientation: {in_degrees(obs[0][3:6])}
            Linear Velocity: {obs[0][6:9]}
            Angular Velocity: {obs[0][9:12]}
            -----------------------------------------------------------------
            Action Space: type {type(action)} value {action}
            Terminated: {terminated}
            Truncated: {truncated}
            -----------------------------------------------------------------
            Policy Architecture: {policy.policy}
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
        if reset and terminated:
            obs, info = test_env.reset(seed=42, options={})

    test_env.close()

    if plot:
        logger.plot_position_and_orientation()
        logger.plot_rpms()
        logger.plot_trajectory()

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
        default=ObS12Stage3,
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
        '--reset',
        default=False,
        type=str2bool,
        help="If you want to reset the environment, every time that the drone achieve the target position"
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
    parser.add_argument(
        '--plot',
        default=False,
        type=str2bool,
        help="If are shown demo plots"
    )
    parser.add_argument(
        '--debug',
        default=False,
        type=str2bool,
        help="Prints debug information"
    )
    parser.add_argument(
        '--filter',
        default=False,
        type=str2bool,
        help="Applies a low pass to the actions coming from the policy"
    )

    run_simulation(**vars(parser.parse_args()))
