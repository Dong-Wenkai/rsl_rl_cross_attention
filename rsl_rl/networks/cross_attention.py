# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """
    A Cross-Attention Block (Transformer style).

    Applies MHA where Q attends to KV. Includes Layer Normalization (Pre-LN style)
    and a residual connection for training stability.
    Assumes inputs Q and KV are already projected to the embedding dimension (D).
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
    ) -> None:
        """Initialize the CrossAttention Block."""
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Layer Normalization (Pre-LN)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        # 2. Multihead Attention
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # Inputs as (Batch, SeqLen, EmbedDim)
        )

        self._output_dim = embed_dim

    @property
    def output_dim(self) -> int:
        """Get the output dimension of the block (D)."""
        return self._output_dim

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # 1. Pre-Layer Normalization
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)

        # 2. Multihead Attention
        # average_attn_weights=False 用于获取每个头的权重以便可视化。
        attn_output, attn_weights = self.mha(
            query=q_norm,
            key=kv_norm,
            value=kv_norm,
            average_attn_weights=False
        )

        # 3. Residual Connection
        output = query + attn_output

        return output, attn_weights

    def init_weights(self) -> None:
        """Initialize weights."""
        pass