#!/usr/bin/env python3
from gym_pybullet_drones.envs import FromScratch
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""######### Learning from scratch starting from random position #########
R(t) = 25 - 20te - 100Bo + 20P - 18we
Target = [0 0 1 0 0 1.5]
##################################################
""")

results = run_learning(environment=FromScratch,
                       learning_id="FROM-SCRATCH-USING-SQUARED-ROOT-ROLL2+PITCH2",
                       continuous_learning=False,
                       stop_on_max_episodes=False,
                       parallel_environments=10,
                       time_steps=int(20e6)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
