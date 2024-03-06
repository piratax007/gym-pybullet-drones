#!/usr/bin/env python3
from gym_pybullet_drones.envs import BasicRewardWithPitchRollPenalty, BasicRewardWithPitchRollPenaltyWithoutR, \
    BasicRewardWithPitchRollPenaltyWithoutWe
from gym_pybullet_drones.experiments.learning_script import run_learning

print("""######### Learning from best with Basic-Reward-and-Pitch-Roll-Penalty Environment #########
R(t) = 25 - 15te - 100Bo + 15Performance - 8R - 13we
Target = [0 0 1 0 0 1.7]
Time steps = 10 Million
Episode Length = 15s""")

results = run_learning(environment=BasicRewardWithPitchRollPenalty,
                       learning_id="KEY-TRAINING-from-best-030324020042-6-terms-reward-10M-ts",
                       continuous_learning=True,
                       stop_on_max_episodes=False,
                       time_steps=10e6
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")

print("""######### Learning from best with Basic-Reward-and-Pitch-Roll-Penalty-without-R Environment #########
R(t) = 25 - 15te - 100Bo + 15Performance - 13we
Target = [0 0 1 0 0 1.7]
Time steps = 10 Million
Episode Length = 15s""")

results = run_learning(environment=BasicRewardWithPitchRollPenaltyWithoutR,
                       learning_id="from-best-030324020042-5-terms-without-R-reward-10M-ts",
                       continuous_learning=True,
                       stop_on_max_episodes=False,
                       time_steps=10e6
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")

print("""######### Learning from best with Basic-Reward-and-Pitch-Roll-Penalty-without-We Environment #########
R(t) = 25 - 15te - 100Bo + 15Performance - 8R
Target = [0 0 1 0 0 1.7]
Time steps = 10 Million
Episode Length = 15s""")

results = run_learning(environment=BasicRewardWithPitchRollPenaltyWithoutWe,
                       learning_id="from-best-030324020042-5-terms-without-We-reward-10M-ts",
                       continuous_learning=True,
                       stop_on_max_episodes=False,
                       time_steps=10e6
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")

print("""######### Learning from scratch with Basic-Reward-And-Pitch-Roll-Penalty Environment #########
R(t) = 25 - 15te - 100Bo + 15Performance - 8R - 13we
Target = [0 0 1 0 0 1.7]
Episodes = 10 Million
Episode Length = 15s""")

results = run_learning(environment=BasicRewardWithPitchRollPenalty,
                       learning_id="from-scratch-6-terms-reward-10M-episodes",
                       parallel_environments=10,
                       episodes=10e6
                       )

print(f"""
################# Learning End ########################
Results: {results}
#######################################################
""")
