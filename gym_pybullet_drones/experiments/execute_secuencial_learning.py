#!/usr/bin/env python3
from gym_pybullet_drones.envs import FromScratchShrink
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""######### Learning to go to a fix position from a random position #########
R(t) = 25 - 20te - 100Bo + 20P - 18we
Target = [0 0 1 0 0 1.5]
##################################################
""")

results = run_learning(environment=FromScratchShrink,
                       learning_id="CONTINUE-RANDOM-START",
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
