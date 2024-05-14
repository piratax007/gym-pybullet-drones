#!/usr/bin/env python3
from gym_pybullet_drones.envs import ObservationSpace12, ObservationSpace72, ObservationSpace72Filter
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""######### From scratch learning observation space 12 #########
R(t) = 25 - 20te - 100Bo + 20P - 18we
Target = [0 0 1 0 0 random]
##################################################
""")

results = run_learning(environment=ObservationSpace12,
                       learning_id="OBS-SIZE-12_FULL-REWARD_TARGET-[0 0 1 0 0 rand]",
                       continuous_learning=False,
                       stop_on_max_episodes=False,
                       parallel_environments=4,
                       time_steps=int(15e6)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")

print("""######### From scratch learning observation space 72 #########
R(t) = 25 - 20te - 100Bo + 20P - 18we
Target = [0 0 1 0 0 random]
##################################################
""")

results = run_learning(environment=ObservationSpace72,
                       learning_id="OBS-SIZE-72_FULL-REWARD_TARGET-[0 0 1 0 0 rand]",
                       continuous_learning=False,
                       stop_on_max_episodes=False,
                       parallel_environments=4,
                       time_steps=int(15e6)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")

print("""######### From scratch learning observation space 72 with filter #########
R(t) = 25 - 20te - 100Bo + 20P - 18we
Target = [0 0 1 0 0 random]
##################################################
""")

results = run_learning(environment=ObservationSpace72Filter,
                       learning_id="OBS-SIZE-72-FILTER_FULL-REWARD_TARGET-[0 0 1 0 0 rand]",
                       continuous_learning=False,
                       stop_on_max_episodes=False,
                       parallel_environments=4,
                       time_steps=int(15e6)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
