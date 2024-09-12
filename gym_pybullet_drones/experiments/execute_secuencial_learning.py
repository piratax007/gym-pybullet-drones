#!/usr/bin/env python3
from gym_pybullet_drones.envs import HoverCrazyflieSim2Real
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""############# Base training #############
Starting = [0 0 0 0 0 0]
Target = [0 0 1 Nan Nan Nan]
##################################################
""")

results = run_learning(environment=HoverCrazyflieSim2Real,
                       learning_id="HOVER_BASE_TRAINING-17",
                       continuous_learning=True,
                       stop_on_max_episodes=False,
                       parallel_environments=4,
                       time_steps=int(80e6)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
