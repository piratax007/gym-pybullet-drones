import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium.spaces import Discrete, Box


class CustomPNNPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space,
            action_space,
            lr_schedule,
            base_columns,
            net_arch=None,
            activation_fn=nn.Tanh,
            **kwargs
    ):
        super(CustomPNNPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs
        )

        self.base_columns = base_columns

        base_feature_dimensions = [
            column.mlp_extractor.latent_dim_pi for column in self.base_columns
        ]

        self.features_dim = self.mlp_extractor.latent_dim_pi

        self.lateral_layers = nn.ModuleList(
            [
                nn.Linear(feature_dim, self.features_dim) for feature_dim in base_feature_dimensions
            ]
        )

        self.new_column = nn.Sequential(
            nn.Linear(self.features_dim * (len(self.base_columns) + 1), 64),
            activation_fn(),
            nn.Linear(64, 64),
            activation_fn(),
        )

        if net_arch is None:
            net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        actor_layers = net_arch["pi"]
        critic_layers = net_arch["vf"]

        actor_layers.insert(0, 12)
        critic_layers.insert(0, 12)

        if isinstance(action_space, Discrete):
            actor_layers.append(action_space.n)
        elif isinstance(action_space, Box):
            actor_layers.append(action_space.shape[0])
        else:
            raise NotImplementedError("Action space type not supported")

    def extract_features(self, obs, deterministic=False):
        return obs

    def forward(self, obs, deterministic=False):
        base_features = [column.extract_features(obs) for column in self.base_columns]
        base_features = torch.cat(
            [
                layer(features) for layer, features in zip(self.lateral_layers, base_features)
            ], dim=1
        )

        new_features = self.extract_features(obs)

        combined_features = torch.cat((base_features, new_features), dim=1)
        self.new_column(combined_features)

        actor_output = self.actor_net(obs)
        critic_output = self.critic_net(obs)

        return actor_output, critic_output

    def _predict(self, obs, deterministic=False):
        action_logits, _ = self.forward(obs, deterministic)
        return action_logits
