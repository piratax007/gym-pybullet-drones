#!/usr/bin/env python3
# from gym_pybullet_drones.envs import BasicRewardWithPitchRollPenalty
from gym_pybullet_drones.envs import HugePenalizationForWe
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""######### Learning from third best #########
R(t) = 25 - 20te - 100Bo + 20P - 18we - 0.0052R
Target = [0 0 1 0 0 1.7]
Time steps = 10M
Episode Length = 8s
##################################################
""")

results = run_learning(environment=HugePenalizationForWe,
                       learning_id="TEST-CHANGING-SQUARES-WITH-MODULE-IN-ANGLES-PENALTIES",
                       continuous_learning=True,
                       stop_on_max_episodes=False,
                       parallel_environments=4,
                       time_steps=int(10e6)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
