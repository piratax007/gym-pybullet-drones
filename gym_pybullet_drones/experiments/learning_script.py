#!/usr/bin/env python3

import os
from datetime import datetime
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_ENV_NAME = HoverAviary
DEFAULT_OUTPUT_FOLDER = 'results'

DEFAULT_OBS = ObservationType('kin')  # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm')  # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1


def run_learning(env_name,
                 learning_id,
                 num_episodes,
                 output_directory=DEFAULT_OUTPUT_FOLDER):

    path_to_results = os.path.join(output_directory, 'save-' + learning_id + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))

    if not os.path.exists(path_to_results):
        os.makedirs(path_to_results + '/')

    train_env = make_vec_env(env_name,
                             n_envs=1,
                             seed=0
                             )
    eval_env = env_name(obs=DEFAULT_OBS, act=DEFAULT_ACT)

    # Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    # Train the model #######################################
    model = PPO('MlpPolicy',
                train_env,
                # tensorboard_log=filename+'/tb/',
                verbose=0,
                device='auto')

    stop_on_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=num_episodes, verbose=1)

    eval_callback = EvalCallback(eval_env,
                                 verbose=0,
                                 best_model_save_path=path_to_results + '/',
                                 log_path=path_to_results + '/',
                                 eval_freq=int(1000),
                                 deterministic=True,
                                 render=False)

    print("""
    ################# Starting learning ###################################
    A learning process is running, please don't close this terminal window.
    #######################################################################
    """)
    model.learn(total_timesteps=int(1e7),  # shorter training in GitHub Actions pytest
                callback=[stop_on_max_episodes, eval_callback],
                log_interval=100)
    print("################# Ending learning ########################")
    model.save(path_to_results + '/final_model.zip')
    return path_to_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single agent RL learning")
    parser.add_argument(
        '--env_name',
        default=DEFAULT_ENV_NAME,
        help='The name of the environment to learn, registered with gym_pybullet_drones'
    )
    parser.add_argument(
        '--output_directory',
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument(
        '--num_episodes',
        default=int(2.5e5),
        type=int,
        help="Number of episodes to run the learning"
    )
    parser.add_argument(
        '--env_parameters',
        default=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
        help="Parameters for the environment to learn"
    )

    results_path = run_learning(**vars(parser.parse_args()))
    print(f" #### The training process has end, the best policy was saved in: {results_path} ####")
