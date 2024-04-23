#!/usr/bin/env python3
from gym_pybullet_drones.envs import Continuous
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""######### Continuous Learning #########
R(t) = 25 - 20te - 100Bo + 20P - 18we
Target = [0 0 1 0 0 1.5]
##################################################
""")

results = run_learning(environment=Continuous,
                       learning_id="CONTINUOUS_SHRINK_EXPLORATION_FROM_04042024220716",
                       continuous_learning=True,
                       stop_on_max_episodes=False,
                       parallel_environments=10,
                       time_steps=int(15e6)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
