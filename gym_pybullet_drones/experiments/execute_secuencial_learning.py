#!/usr/bin/env python3
from gym_pybullet_drones.envs import ObS12Stage1, ObS12Rw5
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""############# From scratch STAGE 1 #############
Starting = [0 0 0 0 0 0]
Target = [0 0 1 None None None]
##################################################
""")

results = run_learning(environment=ObS12Stage1,
                       learning_id="FROM-SCRATCH_STAGE-1-HOVER",
                       continuous_learning=False,
                       stop_on_max_episodes=False,
                       parallel_environments=4,
                       time_steps=int(20e6)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")

print("""############# From scratch FULL TRAINING #############
R(t) = 25 - 20te - 100Bo + 20P - 18W
Starting = [0 0 0 0 0 0]
Target = [0 0 1 0 0 random]
##################################################
""")

results = run_learning(environment=ObS12Rw5,
                       learning_id="FROM-SCRATCH_FULL-TRAINING",
                       continuous_learning=False,
                       stop_on_max_episodes=False,
                       parallel_environments=4,
                       time_steps=int(30e6)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
