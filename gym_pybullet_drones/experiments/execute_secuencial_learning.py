#!/usr/bin/env python3
from gym_pybullet_drones.envs import BasicRewardWithPitchRollPenaltyShrinkingBoundaries
from gym_pybullet_drones.envs import BasicRewardWithPitchRollPenaltyShrinkingBoundariesZeroTargetYaw
from gym_pybullet_drones.envs import BasicRewardWithPitchRollPenaltyShrinkingBoundariesHalfPiYaw
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""######### Learning from best 030324020042 #########
R(t) = 25 - 20te - 100Bo + 20P - 18we
Target = [0 0 1 0 0 1.7]
Time steps = 20M
Episode Length = 15s
##################################################
""")

results = run_learning(environment=BasicRewardWithPitchRollPenaltyShrinkingBoundaries,
                       learning_id="FROM-BEST-030324020042-SHRINKING-BOUNDARIES",
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

print("""######### Learning from best 030324020042 #########
R(t) = 25 - 20te - 100Bo + 20P - 18we
Target = [0 0 1 0 0 0]
Time steps = 20M
Episode Length = 15s
##################################################
""")

results = run_learning(environment=BasicRewardWithPitchRollPenaltyShrinkingBoundariesZeroTargetYaw,
                       learning_id="FROM-BEST-030324020042-SHRINKING-BOUNDARIES-ZERO-TARGET-YAW",
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

print("""######### Learning from scratch zero target yaw #########
R(t) = 25 - 20te - 100Bo + 20P - 18we
Target = [0 0 1 0 0 0]
Episodes = 15M
Episode Length = 15s
##################################################
""")

results = run_learning(environment=BasicRewardWithPitchRollPenaltyShrinkingBoundariesZeroTargetYaw,
                       learning_id="FROM-SCRATCH-ZERO-TARGET-YAW",
                       parallel_environments=4,
                       episodes=int(15e6),
                       time_steps=int(10e8)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")

print("""######### Learning from scratch pi/2 target yaw #########
R(t) = 25 - 20te - 100Bo + 20P - 18we
Target = [0 0 1 0 0 pi/2]
Episodes = 20M
Episode Length = 15s
##################################################
""")

results = run_learning(environment=BasicRewardWithPitchRollPenaltyShrinkingBoundariesHalfPiYaw,
                       learning_id="FROM-SCRATCH-NEW-ACTION-SPACE",
                       parallel_environments=4,
                       episodes=int(20e6),
                       time_steps=int(10e8)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
