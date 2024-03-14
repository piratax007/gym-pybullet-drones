#!/usr/bin/env python3
# from gym_pybullet_drones.envs import BasicRewardWithPitchRollPenalty
from gym_pybullet_drones.envs import BasicRewardWithPitchRollPenaltyShrinkingBoundaries, HugePenalizationForWe
# from gym_pybullet_drones.envs import BasicRewardWithPitchRollPenaltyShrinkingBoundariesZeroTargetYaw
# from gym_pybullet_drones.envs import BasicRewardWithPitchRollPenaltyShrinkingBoundariesHalfPiYaw
from gym_pybullet_drones.experiments.learning_script import run_learning

# print("""######### Learning from best 030324020042 #########
# R(t) = 25 - 20te - 100Bo + 20P - 18we
# Target = [0 0 1 0 0 1.7]
# Time steps = 20M
# Episode Length = 15s
# ##################################################
# """)
#
# results = run_learning(environment=BasicRewardWithPitchRollPenaltyShrinkingBoundaries,
#                        learning_id="TEST-SHRINKING-BOUNDARIES-USING-THIRD-BEST",
#                        continuous_learning=True,
#                        stop_on_max_episodes=False,
#                        parallel_environments=4,
#                        time_steps=int(2e6)
#                        )
#
# print(f"""
# ################# Learning End ########################
# Results: {results}
# #######################################################
# """)

print("""######### Learning from best 030324020042 #########
R(t) = 25 - 20te - 100Bo + 20P - 18we
Target = [0 0 1 0 0 1.7]
Time steps = 20M
Episode Length = 15s
##################################################
""")

results = run_learning(environment=HugePenalizationForWe,
                       learning_id="TEST-SHRINKING-BOUNDARIES-AND-PENALIZING-WE-DIFFERENCES-USING-THIRD-BEST-20M",
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
