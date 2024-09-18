#!/usr/bin/env python3
from gym_pybullet_drones.envs import ObS12Stage3
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""############# Base training #############
Starting = [rand rand rand rand rand rand]
Target = [0 0 1 Nan Nan Nan]
##################################################
""")

results = run_learning(environment=ObS12Stage3,
                       learning_id="CL-TASK3-RANDOM-VELOCITIES",
                       continuous_learning=False,
                       parallel_environments=4,
                       time_steps=int(20e6),
                       stop_on_max_episodes=dict(stop=False, episodes=0),
                       stop_on_reward_threshold=dict(stop=False, threshold=600.),
                       save_checkpoints=dict(save=True, save_frequency=25000)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
