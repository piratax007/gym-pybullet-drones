#!/usr/bin/env python3
from gym_pybullet_drones.envs import HoverSim2Real
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""############# Hover with noise and action buffer #############
Starting = [0 0 0 Nan Nan Nan]
Target = [0 0 1 0 0 Nan]
##################################################
""")

results = run_learning(environment=HoverSim2Real,
                       learning_id="HOVER_NOISE_SIGMA1e-4_ACTION_BUFFER_400-NEURONS",
                       continuous_learning=False,
                       stop_on_max_episodes=False,
                       parallel_environments=10,
                       time_steps=int(10e6)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
