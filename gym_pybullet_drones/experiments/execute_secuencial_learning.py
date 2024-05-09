#!/usr/bin/env python3
from gym_pybullet_drones.envs import ActionsFilter
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""######### ActionsFilter Learning #########
R(t) = 25 - 20te - 100Bo + 20P - 18we
Target = [0 0 1 0 0 1.5]
##################################################
""")

results = run_learning(environment=ActionsFilter,
                       learning_id="NEW_TRAINING_72_OBSERVATION_SPACE_ACTIONS_FILTER",
                       continuous_learning=False,
                       stop_on_max_episodes=False,
                       parallel_environments=4,
                       time_steps=int(10e3)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
