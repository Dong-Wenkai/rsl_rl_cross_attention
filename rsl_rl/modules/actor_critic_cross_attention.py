# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any

from rsl_rl.networks import MLP, CNN, CrossAttention, EmpiricalNormalization

from .actor_critic import ActorCritic


class ActorCriticAttention(ActorCritic):
    """
    Actor-Critic network assembling the pipeline using rsl_rl modules (CNN, MLP, EmpiricalNormalization)
    and the CrossAttention block, matching He et al. (Fig 8B).
    """
    is_recurrent: bool = False

    def __init__(
            self,
            obs: TensorDict,
            obs_groups: dict[str, list[str]],
            num_actions: int,
            # RSL RL 标准化参数
            actor_obs_normalization: bool = True,
            critic_obs_normalization: bool = True,
            # Attention Configuration (Matching the paper defaults)
            embed_dim: int = 64,
            num_heads: int = 16,
            # MLP Configuration
            actor_hidden_dims: list[int] = [256, 128],
            critic_hidden_dims: list[int] = [256, 128],
            activation: str = "elu",
            init_noise_std: float = 1.0,
            # 保持与 ActorCritic 基类兼容所需的参数
            noise_std_type: str = "scalar",
            state_dependent_std: bool = False,
            **kwargs: dict[str, Any],
    ) -> None:
        # 调用父类的 __init__ (即 nn.Module.__init__), 模仿 ActorCriticCNN 的方式
        super(ActorCritic, self).__init__()

        self.obs_groups = obs_groups
        self.embed_dim = embed_dim
        # 存储注意力权重以供可视化（可选）
        self.attn_weights_actor = None
        self.attn_weights_critic = None

        # --------------------------------------------------------------------
        # 1. Dimension Analysis & Normalization Setup
        # --------------------------------------------------------------------

        # A. Actor Inputs (Proprioception + Map Scans)
        actor_proprio_dim, actor_map_dim = self._analyze_inputs(obs, obs_groups.get("policy", []))

        # B. Critic Inputs (Proprioception + Map Scans)
        critic_proprio_dim, critic_map_dim = self._analyze_inputs(obs, obs_groups.get("critic", []))

        # C. Normalization for Proprioception (1D inputs)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(actor_proprio_dim)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(critic_proprio_dim)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        # --------------------------------------------------------------------
        # 2. Encoder Components Definition (Assembling Fig 8B)
        # --------------------------------------------------------------------
        # PPO通常为Actor和Critic使用独立的网络实例。

        # A. Actor Encoder
        self.actor_encoder = self._build_encoder(actor_proprio_dim, actor_map_dim, embed_dim, num_heads, activation)

        # B. Critic Encoder (结构相同，但权重独立)
        self.critic_encoder = self._build_encoder(critic_proprio_dim, critic_map_dim, embed_dim, num_heads, activation)

        # --------------------------------------------------------------------
        # 3. Actor and Critic MLPs (Policy Heads)
        # --------------------------------------------------------------------

        # Actor MLP
        actor_mlp_input_dim = embed_dim + actor_proprio_dim
        self.state_dependent_std = state_dependent_std
        if self.state_dependent_std:
            self.actor = MLP(actor_mlp_input_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(actor_mlp_input_dim, num_actions, actor_hidden_dims, activation)
        print(f"Actor MLP: {self.actor}")

        # Critic MLP
        critic_mlp_input_dim = embed_dim + critic_proprio_dim
        self.critic = MLP(critic_mlp_input_dim, 1, critic_hidden_dims, activation)
        print(f"Critic MLP: {self.critic}")

        # --------------------------------------------------------------------
        # 4. RSL RL Standard Components (Noise)
        # --------------------------------------------------------------------
        # 噪声处理逻辑，完全复制自 ActorCritic/ActorCriticCNN
        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

        self.init_weights()

    def _analyze_inputs(self, obs: TensorDict, group_names: list[str]):
        """Helper to analyze input dimensions."""
        proprio_dim = 0
        map_dim = None

        for name in group_names:
            shape = obs[name].shape
            if len(shape) == 2:  # (B, C) - Proprioception
                proprio_dim += shape[-1]
            elif len(shape) == 4 and shape[1] == 3:  # (B, 3, L, W) - Map Scans
                if map_dim is not None:
                    raise ValueError("Only one map scan input group supported per Actor/Critic.")
                map_dim = shape[2:4]  # (L, W)
            else:
                raise ValueError(f"Invalid observation shape for {name}: {shape}")

        if proprio_dim == 0 or map_dim is None:
            raise ValueError("Requires both proprioception (1D) and map scans (B, 3, L, W) inputs in the group.")

        return proprio_dim, map_dim

    def _build_encoder(self, proprio_dim, map_dim, embed_dim, num_heads, activation):
        """Helper to build the encoder components (CNN, Linear, Attention)."""
        encoder = nn.ModuleDict()

        if embed_dim <= 3: raise ValueError("embed_dim must be > 3.")

        # 1. CNN (K/V Path)
        cnn_cfg = {
            "output_channels": [16, embed_dim - 3],
            "kernel_size": 5,
            "stride": 1,
            "padding": "zeros",  # 自动计算填充以保持维度
            "activation": activation,
            "flatten": False
        }
        # input_channels=1 (Height only)
        encoder['cnn'] = CNN(input_dim=map_dim, input_channels=1, **cnn_cfg)

        # 2. Linear Projection (Q Path)
        encoder['q_proj'] = nn.Linear(proprio_dim, embed_dim)

        # 3. Cross-Attention Module
        encoder['attention'] = CrossAttention(embed_dim=embed_dim, num_heads=num_heads)

        return encoder

    def init_weights(self):
        # 初始化所有组件
        for encoder in [self.actor_encoder, self.critic_encoder]:
            encoder['cnn'].init_weights()
            encoder['attention'].init_weights()
            # 初始化 Q 投影 (Orthogonal)
            nn.init.orthogonal_(encoder['q_proj'].weight, gain=1.0)
            if encoder['q_proj'].bias is not None:
                nn.init.zeros_(encoder['q_proj'].bias)

        # MLP 权重由其自身和噪声初始化逻辑处理

    def _get_features(self, obs: TensorDict, encoder: nn.ModuleDict, normalizer: nn.Module,
                      group_key: str) -> torch.Tensor:
        """Implements the forward pass matching Fig 8B."""
        B = obs.batch_size[0]

        # 1. 准备输入并标准化
        map_scans = None
        proprio_list = []

        # 根据组名提取数据 (区分 Actor/Critic 的输入)
        for name in self.obs_groups[group_key]:
            if len(obs[name].shape) == 2:
                proprio_list.append(obs[name])
            else:
                map_scans = obs[name]

        proprioception_raw = torch.cat(proprio_list, dim=-1)  # (B, d_obs)

        # 应用标准化 (关键步骤)
        proprioception = normalizer(proprioception_raw)

        # 2. K/V 路径 (CNN Encoding)
        # 2a. 提取高度 (Z-coordinate, index 2). (B, 1, L, W)
        height_map = map_scans[:, 2:3, :, :]
        # 2b. CNN 处理. (B, D-3, L, W)
        cnn_features = encoder['cnn'](height_map)
        # 2c. 拼接 CNN 特征和原始 XYZ 坐标. (B, D, L, W)
        combined_features = torch.cat((cnn_features, map_scans), dim=1)
        # 2d. 重塑为序列 (B, L*W, D)
        keys_values = combined_features.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)

        # 3. Q 路径 (Linear Projection) - 使用标准化后的本体感知
        query_flat = encoder['q_proj'](proprioception)  # (B, D)
        query = query_flat.view(B, 1, self.embed_dim)  # N=1

        # 4. 交叉注意力模块 (MHA)
        attn_output, attn_weights = encoder['attention'](query, keys_values)

        # 存储权重（可选）
        if group_key == "policy":
            self.attn_weights_actor = attn_weights
        else:
            self.attn_weights_critic = attn_weights

        # 5. 最终拼接 (Fig 8B: Top Concat) - 使用标准化后的本体感知
        map_encoding = attn_output.view(B, -1)
        features = torch.cat((map_encoding, proprioception), dim=-1)

        return features

    # --- RSL RL Interface Methods ---

    # 覆盖基类的 _update_distribution，实现完整的噪声处理逻辑 (复制自 ActorCritic.py)
    def _update_distribution(self, features: torch.Tensor) -> None:
        if self.state_dependent_std:
            # Compute mean and standard deviation
            mean_and_std = self.actor(features)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            # Compute mean
            mean = self.actor(features)
            # Compute standard deviation
            if self.noise_std_type == "scalar":
                std = self.std.expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Create distribution
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        # 获取 Actor 特征 (使用 Actor 的编码器、标准化器和观测组 "policy")
        features = self._get_features(obs, self.actor_encoder, self.actor_obs_normalizer, "policy")
        self._update_distribution(features)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        features = self._get_features(obs, self.actor_encoder, self.actor_obs_normalizer, "policy")
        if self.state_dependent_std:
            return self.actor(features)[..., 0, :]
        else:
            return self.actor(features)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        # 获取 Critic 特征 (使用 Critic 的编码器、标准化器和观测组 "critic")
        features = self._get_features(obs, self.critic_encoder, self.critic_obs_normalizer, "critic")
        return self.critic(features)

    # 覆盖 update_normalization 以处理本体感知输入
    def update_normalization(self, obs: TensorDict) -> None:
        # 更新 Actor 标准化器
        if self.actor_obs_normalization:
            # 提取 Actor 的原始本体感知数据 (仅 1D 数据)
            proprio_list = [obs[name] for name in self.obs_groups["policy"] if len(obs[name].shape) == 2]
            if proprio_list:
                actor_proprio = torch.cat(proprio_list, dim=-1)
                self.actor_obs_normalizer.update(actor_proprio)

        # 更新 Critic 标准化器
        if self.critic_obs_normalization:
            # 提取 Critic 的原始本体感知数据 (仅 1D 数据)
            proprio_list = [obs[name] for name in self.obs_groups["critic"] if len(obs[name].shape) == 2]
            if proprio_list:
                critic_proprio = torch.cat(proprio_list, dim=-1)
                self.critic_obs_normalizer.update(critic_proprio)

    # (其他方法如 get_actions_log_prob, action_mean, entropy 等由基类 ActorCritic 提供)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.attn_weights_actor = None
        self.attn_weights_critic = None