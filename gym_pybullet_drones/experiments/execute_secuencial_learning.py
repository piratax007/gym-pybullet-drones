#!/usr/bin/env python3
from gym_pybullet_drones.envs import FromScratch
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""######### Learning from third best #########
R(t) = 25 - 20te - 100Bo + 20P - 18we
Target = [0 0 1 0 0 np.pi/2]
Time steps = 10M
Episode Length = 8s
##################################################
""")

results = run_learning(environment=FromScratch,
                       learning_id="FROM-SCRATCH-10M-EPISODES",
                       parallel_environments=4,
                       time_steps=int(10e8),
                       episodes=int(10e6)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
