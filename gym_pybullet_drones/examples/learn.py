"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin')  # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm')  # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False


def run(output_folder=DEFAULT_OUTPUT_FOLDER,
        gui=DEFAULT_GUI, plot=True,
        colab=DEFAULT_COLAB,
        record_video=DEFAULT_RECORD_VIDEO,
        local=True):

    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    train_env = make_vec_env(HoverAviary,
                             env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
                             n_envs=1,
                             seed=0
                             )
    eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)

    # Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    # Train the model #######################################
    model = PPO('MlpPolicy',
                train_env,
                # tensorboard_log=filename+'/tb/',
                verbose=1)

    stop_on_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=int(2.5e5), verbose=1)

    eval_callback = EvalCallback(eval_env,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)
    model.learn(total_timesteps=int(1e7) if local else int(1e2),  # shorter training in GitHub Actions pytest
                callback=[stop_on_max_episodes, eval_callback],
                log_interval=100)

    # Save the model ########################################
    model.save(filename+'/final_model.zip')
    print(filename)

    if local:
        input("Press Enter to continue...")

    if os.path.isfile(filename+'/best_model.zip'):
        model = PPO.load(filename+'/best_model.zip')
    else:
        print("[ERROR]: no model under the specified path", filename)

    test_env = HoverAviary(gui=gui,
                           obs=DEFAULT_OBS,
                           act=DEFAULT_ACT,
                           record=record_video)

    logger = Logger(
        logging_freq_hz=int(test_env.CTRL_FREQ),
        num_drones=1,
        output_folder=output_folder,
        colab=colab
    )

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()

    log_reward = []

    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs,
                                        deterministic=True
                                        )
        obs, reward, terminated, truncated, info = test_env.step(action)
        log_reward.append(reward)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print(f"""
        #################################################################
        Observation Space:
        Position: {obs[0][0:3]}
        Orientation: {obs[0][3:6]}
        Linear Velocity: {obs[0][6:9]}
        Angular Velocity: {obs[0][9:12]}
        -----------------------------------------------------------------
        Action Space: {action}
        Reward: {reward}
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
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()
        logger.plot_instantaneous_reward(filename, log_reward)


if __name__ == '__main__':
    # Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument(
        '--gui',
        default=DEFAULT_GUI,
        type=str2bool,
        help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument(
        '--record_video',
        default=DEFAULT_RECORD_VIDEO,
        type=str2bool,
        help='Whether to record a video (default: False)', metavar='')
    parser.add_argument(
        '--output_folder',
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument(
        '--colab',
        default=DEFAULT_COLAB,
        type=bool,
        help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
