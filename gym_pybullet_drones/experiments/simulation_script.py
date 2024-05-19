#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
from stable_baselines3 import PPO
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool, FIRFilter
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.envs import ObS12Stage2


def in_degrees(angles):
    return list(map(lambda angle: angle * 180 / np.pi, angles))


def get_policy(policy_path):
    if os.path.isfile(policy_path + '/best_model.zip'):
        return PPO.load(policy_path + '/best_model.zip')

    raise Exception("[ERROR]: no model under the specified path", policy_path)


def run_simulation(
        policy_path,
        test_env, gui=True,
        record_video=False,
        reset=False,
        save=False,
        plot=False,
        comment=""
):
    policy = get_policy(policy_path)

    test_env = test_env(gui=gui,
                        obs=ObservationType('kin'),
                        act=ActionType('rpm'),
                        initial_xyzs=np.array([[1, 1, 0]]),
                        initial_rpys=np.array([[0, 0, 0.4]]),
                        record=record_video)

    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1,
        output_folder=policy_path,
        colab=False
    )

    obs, info = test_env.reset(seed=42, options={})
    log_reward = []
    simulation_length = (test_env.EPISODE_LEN_SEC + 12) * test_env.CTRL_FREQ

    start = time.time()

    firfilter = FIRFilter()
    for _ in range(firfilter.buffer_size):
        firfilter.buffer.append(np.zeros((1, 4)))

    for i in range(simulation_length):
        # if i < (simulation_length / 5):
        #     x_target = 0
        #     y_target = 0
        #     z_target = 1
        #     yaw_target = 0
        # elif i < 2 * (simulation_length / 5):
        #     x_target = 1
        #     y_target = 0
        #     z_target = 0.5
        #     yaw_target = -0.5
        # elif i < 3 * (simulation_length / 5):
        #     x_target = 1
        #     y_target = 1
        #     z_target = 0.5
        #     yaw_target = -1.0
        # elif i < 4 * (simulation_length / 5):
        #     x_target = 0
        #     y_target = 1
        #     z_target = 1
        #     yaw_target = -1.5
        # else:
        #     x_target = -1
        #     y_target = 1
        #     z_target = 1
        #     yaw_target = -2.0
        #
        # obs[0][0] += 1
        # obs[0][1] -= 1
        # obs[0][2] += 1 - z_target
        # obs[0][5] += yaw_target
        # obs[0][0] -= test_env.INIT_XYZS[0][0]
        # obs[0][1] -= test_env.INIT_XYZS[0][1]
        action, _states = policy.predict(obs,
                                         deterministic=True
                                         )

        # action = firfilter.filter_actions(action)

        obs, reward, terminated, truncated, info = test_env.step(action)
        log_reward.append(reward)
        actions = test_env._getDroneStateVector(0)[16:20]
        actions2 = actions.squeeze()
        obs2 = obs.squeeze()
        # print(f"""
        # #################################################################
        # Observation Space:
        # Position: {obs[0][0:3]}
        # Orientation: {in_degrees(obs[0][3:6])}
        # Linear Velocity: {obs[0][6:9]}
        # Angular Velocity: {obs[0][9:12]}
        # -----------------------------------------------------------------
        # Action Space: type {type(action)} value {action}
        # Terminated: {terminated}
        # Truncated: {truncated}
        # -----------------------------------------------------------------
        # Policy Architecture: {policy.policy}
        # #################################################################
        # """)

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
#         logger.plot_angular_velocities()

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
        default=ObS12Stage2,
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

    run_simulation(**vars(parser.parse_args()))
