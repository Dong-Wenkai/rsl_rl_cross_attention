# Filename: modules/actor_critic_attention.py

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn


# 假设这些模块可以被正确导入。请根据你的项目结构调整导入路径。
# 例如:
# try:
#     # 导入 rsl_rl 提供的 MLP
#     from rsl_rl.networks import MLP
#     # 导入我们刚定义的模块
#     from rsl_rl.networks.cross_attention import CrossAttention, SpatialFeatureEncoder
# except ImportError:
#      print("Warning: Cannot import MLP, CrossAttention or SpatialFeatureEncoder.")


class ActorCriticCrossAttention(nn.Module):
    """
    Actor-Critic network implementing the architecture described in
    "Attention-based map encoding for learning generalized legged locomotion" (Fig 8B).
    """
    is_recurrent: bool = False

    def __init__(
            self,
            obs: TensorDict,
            obs_groups: dict[str, list[str]],
            num_actions: int,
            # Attention Encoder Configuration (Matching the paper defaults)
            embed_dim: int = 64,
            num_heads: int = 16,
            # MLP Configuration
            actor_hidden_dims: list[int] = [256, 128],  # 根据需要调整
            critic_hidden_dims: list[int] = [256, 128],
            activation: str = "elu",
            init_noise_std: float = 1.0,
            **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCriticAttention.__init__ got unexpected arguments, which will be ignored: " + str(kwargs)
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.embed_dim = embed_dim
        self.attn_weights = None  # Store attention weights for visualization

        # --------------------------------------------------------------------
        # 1. Dimension Analysis
        # --------------------------------------------------------------------

        # A. Proprioception (d_obs)
        proprio_dim = 0
        for obs_name in obs_groups.get("proprioception", []):
            proprio_dim += obs[obs_name].shape[-1]
        if proprio_dim == 0:
            raise ValueError("Requires 'proprioception' observation group.")
        self.proprio_dim = proprio_dim

        # B. Map Scans (L, W, 3)
        map_scan_names = obs_groups.get("map_scans", [])
        if not map_scan_names or len(map_scan_names) > 1:
            raise ValueError("Requires exactly one 'map_scans' input tensor.")

        # 存储观测名称以便在 forward 中使用
        self.map_scan_name = map_scan_names[0]
        map_shape = obs[self.map_scan_name].shape

        # Expected shape (B, 3, L, W) for CNN processing (Channels-First)
        if len(map_shape) != 4 or map_shape[1] != 3:
            raise ValueError(f"Expected map_scans shape (B, 3, L, W), but got {map_shape}")

        # --------------------------------------------------------------------
        # 2. Encoder Initialization (Fig 8B Encoder)
        # --------------------------------------------------------------------

        # A. Spatial Encoder (Keys/Values)
        self.spatial_encoder = SpatialFeatureEncoder(
            embed_dim=embed_dim,
            activation=activation,
        )

        # B. Proprioception Projection (Query)
        self.q_projection = nn.Linear(proprio_dim, embed_dim)

        # C. Cross-Attention (MHA)
        self.attention = CrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        # The Query sequence length (N) is fixed at 1 in this architecture.
        self.query_seq_len = 1

        print(f"Attention Encoder initialized. D={embed_dim}, Heads={num_heads}, d_obs={proprio_dim}")

        # --------------------------------------------------------------------
        # 3. Actor and Critic MLPs (Fig 8B Policy)
        # --------------------------------------------------------------------

        # Input dimension is the concatenation of Attention Output (D) and original Proprioception (d_obs)
        mlp_input_dim = embed_dim + proprio_dim

        self.actor = MLP(mlp_input_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # Critic shares the same input structure
        self.critic = MLP(mlp_input_dim, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # --------------------------------------------------------------------
        # 4. RSL RL Standard Components
        # --------------------------------------------------------------------

        # Observation normalization (Not explicitly used in the paper's architecture)
        self.actor_obs_normalizer = nn.Identity()
        self.critic_obs_normalizer = nn.Identity()

        # Action noise (Using fixed standard deviation)
        self.state_dependent_std = False
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args(False)

        self.init_weights()

    def init_weights(self):
        # Initialize network components
        self.spatial_encoder.init_weights()
        self.attention.init_weights()

        # Initialize Q projection (Orthogonal initialization)
        nn.init.orthogonal_(self.q_projection.weight, gain=1.0)
        if self.q_projection.bias is not None:
            nn.init.zeros_(self.q_projection.bias)

        # Initialize MLPs (Assuming MLP class has init_weights method)
        self.actor.init_weights(scales=1.0)
        self.critic.init_weights(scales=1.0)

    def _get_features(self, obs: TensorDict) -> torch.Tensor:
        """Implements the forward pass matching Fig 8B."""
        B = obs.batch_size[0]

        # 1. Prepare Inputs
        # Map Scans (B, 3, L, W)
        map_scans = obs[self.map_scan_name]

        # Proprioception (B, d_obs)
        proprio_list = [obs[obs_name] for obs_name in self.obs_groups["proprioception"]]
        proprioception = torch.cat(proprio_list, dim=-1)

        # 2. Generate Keys/Values (Spatial Encoding)
        # Output: (B, L*W, D)
        keys_values = self.spatial_encoder(map_scans)

        # 3. Generate Query (Proprioception Projection)
        query_flat = self.q_projection(proprioception)  # (B, D)
        # Reshape to (B, N, D) where N=1
        query = query_flat.view(B, self.query_seq_len, self.embed_dim)

        # 4. Cross-Attention
        # Output: (B, N, D), Weights: (B, H, N, L*W)
        # Note: We do not add LayerNorm or Residual connections here to strictly follow Fig 8B.
        attn_output, self.attn_weights = self.attention(query, keys_values)

        # 5. Final Concatenation (Fig 8B: Concat)
        # Flatten attention output: (B, D)
        attn_output_flat = attn_output.view(B, -1)
        # Concatenate with original proprioception (B, D + d_obs)
        features = torch.cat((attn_output_flat, proprioception), dim=-1)

        return features

    # --- Standard ActorCritic Properties/Methods (Required by PPO) ---
    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, features: torch.Tensor) -> None:
        mean = self.actor(features)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    # --- RSL RL Interface Methods ---
    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        features = self._get_features(obs)
        # Note: Normalizer is Identity here, included for compatibility
        features = self.actor_obs_normalizer(features)
        self._update_distribution(features)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        features = self._get_features(obs)
        features = self.actor_obs_normalizer(features)
        return self.actor(features)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        # Actor and Critic share the encoder structure
        features = self._get_features(obs)
        features = self.critic_obs_normalizer(features)
        return self.critic(features)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    # --- Compatibility Methods (Mimicking ActorCritic.py interface) ---
    def reset(self, dones: torch.Tensor | None = None) -> None:
        # Clear stored attention weights on reset
        self.attn_weights = None

    def forward(self) -> NoReturn:
        raise NotImplementedError

    def update_normalization(self, obs: TensorDict) -> None:
        # If normalization were implemented, it would be updated here.
        pass

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True