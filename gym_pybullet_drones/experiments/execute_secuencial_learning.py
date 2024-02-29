#!/usr/bin/env python3

from gym_pybullet_drones.envs import SimBasicReward, RewardRPMsAndC9, RewardRollPitchRPMsC9, RewardRErEeBi
from gym_pybullet_drones.experiments.learning_script import run_learning

# print("######### Learning with RewardRErEeBi Environment #########")
# # R(t) = 25 - 15te - 100Bo - 15Er -10Ee - 2R + 100Bi
# # Target = [0 0 1 0 0 1.7]
# run_learning(env_name=RewardRErEeBi,
#              learning_id="RewardRErEeBi",
#              num_episodes=int(2.5e6),
#              )
#
# print("######### Learning with RewardRollPitchRPMsC9 Environment #########")
# # R(t) = 25 - 15te - 100Bo -15Er - 2R + 100Bi
# # Target = [0 0 1 0 0 1.7]
# run_learning(env_name=RewardRollPitchRPMsC9,
#              learning_id="RewardRollPitchRPMsC9",
#              num_episodes=int(2.5e6),
#              )
#
# print("######### Learning with RewardRPMsAndC9 Environment #########")
# # R(t) = 25 - 15te - 100Bo - 2R + 100Bi
# # Target = [0 0 1 0 0 1.7]
# run_learning(env_name=RewardRPMsAndC9,
#              learning_id="RewardRPMsAndC9",
#              num_episodes=int(2.5e6),
#              )

print("######### Learning with SimBasicReward Environment #########")
# R(t) = 25 - 15te - 100Bo
# Target = [0 0 1 0 0 1.7]
run_learning(env_name=SimBasicReward,
             learning_id="SimBasicReward",
             num_episodes=int(5e6),
             )
