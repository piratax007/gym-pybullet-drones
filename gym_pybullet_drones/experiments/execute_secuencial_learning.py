#!/usr/bin/env python3
# from gym_pybullet_drones.envs import BasicRewardWithPitchRollPenalty
from gym_pybullet_drones.envs import HugePenalizationForWe
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""######### Learning from third best #########
R(t) = 25 - 20te - 100Bo + 20P - 18we - 0.0104R
Target = [0 0 1 0 0 1.7]
Time steps = 20M
Episode Length = 15s
##################################################
""")

results = run_learning(environment=HugePenalizationForWe,
                       learning_id="TEST-WITH-RPMS-DIFFERENCE-10M",
                       continuous_learning=True,
                       stop_on_max_episodes=False,
                       parallel_environments=4,
                       time_steps=int(2e6)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")

# print("""######### Learning from third best #########
# R(t) = 25 - 20te - 100Bo + 20P - 20we
# Target = [0 0 1 0 0 1.7]
# Episodes = 20M
# Episode Length = 15s
# ##################################################
# """)
#
# results = run_learning(environment=HugePenalizationForWe,
#                        learning_id="FROM-SCRATCH-SHRINKING-BOUNDARIES-AND-PENALIZING-WE-DIFFERENCES-USING-THIRD-BEST",
#                        parallel_environments=4,
#                        episodes=int(20e6),
#                        time_steps=int(10e8)
#                        )
#
# print(f"""
# ################# Learning End ########################
# Results: {results}
# #######################################################
# """)
