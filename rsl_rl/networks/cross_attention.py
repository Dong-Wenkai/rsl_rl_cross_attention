# Filename: networks/cross_attention.py

# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

# 假设 rsl_rl.utils 可以被导入
try:
    from rsl_rl.utils import resolve_nn_activation
except ImportError:
    print("Warning: Cannot import rsl_rl.utils. Using fallback activation resolution.")
    def resolve_nn_activation(activation: str | None):
        if activation == "elu": return nn.ELU()
        if activation == "relu": return nn.ReLU()
        return nn.ELU() # Default fallback

class CrossAttention(nn.Module):
    """
    A wrapper around PyTorch's MultiheadAttention (MHA).
    Performs cross-attention where Q attends to KV pairs.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Multihead Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True # Inputs as (Batch, SeqLen, EmbedDim)
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            query: The query tensor (Batch, SeqLen_Q, D).
            key_value: The key/value tensor (Batch, SeqLen_KV, D).

        Returns:
            The attended output tensor (Batch, SeqLen_Q, D).
            Attention weights (B, num_heads, SeqLen_Q, SeqLen_KV) - useful for visualization.
        """
        # In this architecture, K and V are the same (spatial features)
        # We request average_attn_weights=False to get per-head weights for visualization.
        attn_output, attn_weights = self.attention(
            query=query,
            key=key_value,
            value=key_value,
            average_attn_weights=False
        )

        return attn_output, attn_weights

    def init_weights(self) -> None:
        # Default initialization is usually sufficient.
        pass


class SpatialFeatureEncoder(nn.Module):
    """
    Encodes spatial map scans into features suitable for attention (Keys/Values).
    Implements the CNN pathway described in He et al. (Fig 8B).

    Process:
    1. Extract height (Z) from 3D map scans (XYZ).
    2. Process height with a 2-layer CNN (16 channels, then D-3 channels).
    3. Concatenate CNN output with the original 3D coordinates (XYZ).
    """
    def __init__(
        self,
        embed_dim: int,
        activation: str = "elu",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # The final CNN layer must output D-3 channels.
        if embed_dim <= 3:
            raise ValueError(f"embed_dim must be greater than 3, but got {embed_dim}")
        final_cnn_channels = embed_dim - 3

        activation_mod = resolve_nn_activation(activation)

        # As per the paper (Page 13): Kernel size 5, zero padding (padding=2 for k=5).
        kernel_size = 5
        padding = 2

        # CNN structure: 1 (Height) -> 16 -> D-3
        self.cnn = nn.Sequential(
            # Layer 1
            nn.Conv2d(1, 16, kernel_size=kernel_size, padding=padding, padding_mode='zeros'),
            activation_mod,
            # Layer 2
            nn.Conv2d(16, final_cnn_channels, kernel_size=kernel_size, padding=padding, padding_mode='zeros'),
            # Typically an activation function is used after the final CNN layer
            activation_mod
        )

    def forward(self, map_scans: torch.Tensor) -> torch.Tensor:
        """
        Args:
            map_scans: 3D coordinates of map points (B, 3, L, W).
                       Assumes channels are ordered [X, Y, Z].

        Returns:
            Map features (B, L*W, D).
        """
        B, C, L, W = map_scans.shape
        if C != 3:
             raise ValueError(f"Expected 3 channels (XYZ) for map_scans, but got {C}")

        # 1. Extract height (Z-coordinate). Assuming Z is the last channel (index 2).
        # Output: (B, 1, L, W)
        height_map = map_scans[:, 2:3, :, :]

        # 2. Process height with CNN
        # Output: (B, D-3, L, W)
        cnn_features = self.cnn(height_map)

        # 3. Concatenate CNN features with original 3D coordinates (Fig 8B: Concat)
        # Output: (B, D, L, W)
        combined_features = torch.cat((cnn_features, map_scans), dim=1)

        # 4. Reshape for attention (B, D, L, W) -> (B, L, W, D) -> (B, L*W, D)
        # Permute to put channels last, then flatten spatial dimensions.
        map_features = combined_features.permute(0, 2, 3, 1).reshape(B, L * W, self.embed_dim)

        return map_features

    def init_weights(self) -> None:
        """Initialize the weights of the CNN using Kaiming initialization."""
        for module in self.cnn:
            if isinstance(module, nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)