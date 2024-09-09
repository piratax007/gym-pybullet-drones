#!/usr/bin/env python3

import os
from datetime import datetime
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_OUTPUT_FOLDER = 'results'

DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')


def results_directory(base_directory, results_id):
    path = os.path.join(base_directory, 'save-' + results_id + '-' + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))

    if not os.path.exists(path):
        os.makedirs(path + '/')

    return str(path)


def get_ppo_model(environment, path, reuse_model=False):
    if reuse_model:
        return PPO.load(path=path,
                        device='cuda',
                        env=environment,
                        force_reset=True)

    return PPO('MlpPolicy',
               environment,
               tensorboard_log=path + '/tb/',
               batch_size=128,
               verbose=0,
               device='cuda')


def callbacks(episodes, evaluation_environment, parallel_environments, path_to_results, stop_on_max_episodes):
    eval_callback = EvalCallback(evaluation_environment,
                                 verbose=0,
                                 best_model_save_path=path_to_results + '/',
                                 log_path=path_to_results + '/',
                                 eval_freq=int(10000 / parallel_environments),
                                 deterministic=True,
                                 render=False)

    if stop_on_max_episodes:
        stop_on_max_episodes = StopTrainingOnMaxEpisodes(int(episodes / parallel_environments), verbose=1)
        callback_list = [stop_on_max_episodes, eval_callback]
    else:
        callback_list = [eval_callback]
    return callback_list


def run_learning(environment,
                 learning_id,
                 continuous_learning=False,
                 stop_on_max_episodes=True,
                 parallel_environments=4,
                 time_steps=10e7,
                 episodes=int(1e6),
                 output_directory=DEFAULT_OUTPUT_FOLDER):
    path_to_results = results_directory(output_directory, learning_id)

    learning_environment = make_vec_env(environment,
                                        n_envs=parallel_environments
                                        )
    evaluation_environment = make_vec_env(environment,
                                          n_envs=parallel_environments
                                          )

    model = get_ppo_model(learning_environment,
                          'continuous_learning/best_model.zip' if continuous_learning else path_to_results,
                          continuous_learning)

    callback_list = callbacks(episodes, evaluation_environment, parallel_environments, path_to_results,
                              stop_on_max_episodes)

    model.learn(total_timesteps=int(time_steps),
                callback=callback_list,
                log_interval=1,
                progress_bar=True)

    model.save(path_to_results + '/final_model.zip')

    return path_to_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Single agent RL learning")
    parser.add_argument(
        '--env_name',
        help='The name of the environment to learn, registered with gym_pybullet_drones'
    )
    parser.add_argument(
        '--output_directory',
        default=DEFAULT_OUTPUT_FOLDER,
        type=str,
        help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument(
        '--env_parameters',
        default=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT),
        help="Parameters for the environment to learn"
    )

    results_path = run_learning(**vars(parser.parse_args()))
    print(f" #### The training process has end, the best policy was saved in: {results_path} ####")
