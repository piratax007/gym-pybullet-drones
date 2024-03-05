#!/usr/bin/env python3
from gym_pybullet_drones.envs import BasicRewardWithPitchRollPenalty
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""######### Learning with Basic-Reward-With-Pitch-Roll-Penalty Environment #########
R(t) = 25 - 15te - 100Bo + 15Per
Target = [0 0 1 0 0 1.7]
Episodes = 1 Million
Episode Length = 8s""")

results = run_learning(environment=BasicRewardWithPitchRollPenalty,
                       learning_id="from-scratch-1M-episodes",
                       episodes=1e6,
                       parallel_environments=10
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")

print("""######### Learning with Basic-Reward-With-Pitch-Roll-Penalty Environment #########
R(t) = 25 - 15te - 100Bo + 15Per
Target = [0 0 1 0 0 1.7]
Episodes = 2.5 Million
Episode Length = 8s""")

results = run_learning(environment=BasicRewardWithPitchRollPenalty,
                       learning_id="from-scratch-2hM-episodes",
                       episodes=2.5e6,
                       parallel_environments=10
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")

print("""######### Learning with Basic-Reward-With-Pitch-Roll-Penalty Environment #########
R(t) = 25 - 15te - 100Bo + 15Per
Target = [0 0 1 0 0 1.7]
Episodes = 2.5 Million
Episode Length = 8s""")

results = run_learning(environment=BasicRewardWithPitchRollPenalty,
                       learning_id="from-old-best",
                       continuous_learning=True,
                       stop_on_max_episodes=False,
                       time_steps=2e6
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
