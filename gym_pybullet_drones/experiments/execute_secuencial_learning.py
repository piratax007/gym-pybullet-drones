#!/usr/bin/env python3
from gym_pybullet_drones.envs import FromScratch, FromScratchShrink
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""######### Learning from best from scratch #########
R(t) = 25 - 20te - 100Bo + 20P - 18we
Target = [0 0 1 0 0 np.pi/2]
##################################################
""")

results = run_learning(environment=FromScratchShrink,
                       learning_id="CONTINUE-SHRINK",
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

print("""######### Learning from scratch #########
R(t) = 25 - 20te - 100Bo + 20P - 18we
Target = [0 0 1 0 0 np.pi/2]
##################################################
""")

results = run_learning(environment=FromScratch,
                       learning_id="FROM_SCRATCH_PARALLEL_10_20M_TS",
                       stop_on_max_episodes=False,
                       parallel_environments=10,
                       time_steps=int(20e6)
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
