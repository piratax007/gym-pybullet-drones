#!/usr/bin/env python3
from gym_pybullet_drones.envs import BasicRewardWithPitchRollPenalty
# from gym_pybullet_drones.envs.BasicRewardWithPerformWeRPMsPenalties import BasicRewardWithPerformWeRPMsPenalties
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""######### Learning from best 030324020042 #########
R(t) = 25 - 20te - 100Bo + 20Per - 18we
Target = [0 0 1 0 0 1.7]
episodes = 15M
Episode Length = 15s
##################################################
""")

results = run_learning(environment=BasicRewardWithPitchRollPenalty,
                       learning_id="FROM-BEST-030324020042-PERFORMANCE-IMPROVING-CLOSED",
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

# print("""######### Learning from best 030324020042 #########
# R(t) = 25 - 20te - 100Bo + 20Per - 18we - 0.5||R||
# Target = [0 0 1 0 0 1.7]
# Time steps = 10M
# Episode Length = 15s
# ##################################################
# """)
#
# results = run_learning(environment=BasicRewardWithPerformWeRPMsPenalties,
#                        learning_id="FROM-BEST-030324020042-PERFORMANCE-WE-AND-RPMs",
#                        continuous_learning=True,
#                        stop_on_max_episodes=False,
#                        parallel_environments=4,
#                        time_steps=int(10e6)
#                        )
#
# print(f"""
# ################# Learning End ########################
# Results: {results}
# #######################################################
# """)
