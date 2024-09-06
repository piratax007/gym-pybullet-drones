#!/usr/bin/env python3
from gym_pybullet_drones.envs import ObS12Stage1
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""############# Base training #############
Starting = [0 0 0 Nan Nan Nan]
Target = [0 0 1 Nan Nan Nan]
##################################################
""")

results = run_learning(environment=ObS12Stage1,
                       learning_id="HOVER_BASE_TRAINING",
                       continuous_learning=False,
                       stop_on_max_episodes=False,
                       parallel_environments=100,
                       time_steps=int(20e6)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
