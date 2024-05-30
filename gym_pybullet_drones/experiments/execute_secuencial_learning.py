#!/usr/bin/env python3
from gym_pybullet_drones.envs import ObS12Stage2
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""############# From scratch STAGE 2 #############
Starting = [0 0 0 0 0 0]
Target = [0 0 1 0 0 0]
##################################################
""")

results = run_learning(environment=ObS12Stage2,
                       learning_id="CL-TASK2",
                       continuous_learning=True,
                       stop_on_max_episodes=False,
                       parallel_environments=4,
                       time_steps=int(20e6)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
