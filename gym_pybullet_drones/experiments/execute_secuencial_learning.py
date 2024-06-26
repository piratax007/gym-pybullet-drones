#!/usr/bin/env python3
from gym_pybullet_drones.envs import ObS12Stage3
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""############# From scratch STAGE 2 #############
Starting = [rand rand rand Nan Nan Nan]
Target = [0 0 1 0 0 rand]
##################################################
""")

results = run_learning(environment=ObS12Stage3,
                       learning_id="CL-new_TASK3",
                       continuous_learning=True,
                       stop_on_max_episodes=False,
                       parallel_environments=4,
                       time_steps=int(16e6)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
