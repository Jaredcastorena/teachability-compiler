"""A small decoder-only Transformer for real learner measurements."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DecoderConfig:
    """Configuration for :class:`TinyDecoder`.

    The default model is approximately 9.9M trainable parameters for the 98-token printable
    character vocabulary used by ``real.tasks``.
    """

    vocab_size: int
    d_model: int = 320
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1280
    max_seq_len: int = 64
    dropout: float = 0.0


class TinyDecoder(nn.Module):
    """Pre-norm causal decoder Transformer with tied input/output embeddings."""

    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        if config.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if config.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([_DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return next-token logits with shape ``[batch, time, vocab_size]``."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        positions = torch.arange(seq_len, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        hidden = self.dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        return self.lm_head(hidden)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class _DecoderBlock(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = _CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return hidden


class _CausalSelfAttention(nn.Module):
    def __init__(self, config: DecoderConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = hidden.shape
        query, key, value = self.qkv(hidden).chunk(3, dim=-1)

        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.residual_dropout(self.output(attended))


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
