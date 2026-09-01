from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn

from .common import FeatureEmbedding, MLP


class MetadataEmbedding(nn.Module):
    def __init__(self, dims: Mapping[str, int], embedding_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.names = tuple(dims)
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(int(width) + 1, embedding_dim, padding_idx=0)
            for name, width in dims.items()
        })
        self.projection = nn.Linear(len(self.names) * embedding_dim, hidden_dim)

    def forward(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        parts = [self.embeddings[name](values[name].long()) for name in self.names]
        return torch.tanh(self.projection(torch.cat(parts, dim=-1)))


class CandidateAwareDIN(nn.Module):
    """Candidate-conditioned attention over strictly earlier positive events."""

    def __init__(
        self, field_dims: Sequence[int], metadata_dims: Mapping[str, int],
        embedding_dim: int = 16, hidden_dim: int = 64, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.field_dims = [int(value) for value in field_dims]
        self.metadata = MetadataEmbedding(metadata_dims, embedding_dim, hidden_dim)
        self.base = FeatureEmbedding(self.field_dims, embedding_dim)
        self.attention = MLP(hidden_dim * 4, (64, 32), dropout=dropout, output_dim=1)
        self.scorer = MLP(
            len(self.field_dims) * embedding_dim + hidden_dim * 2,
            (128, 64), dropout=dropout, output_dim=1,
        )

    def forward(
        self, candidate_x: torch.Tensor, history: Mapping[str, torch.Tensor],
        history_mask: torch.Tensor, candidate_metadata: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        historical = self.metadata(history)
        candidate = self.metadata(candidate_metadata)
        expanded = candidate.unsqueeze(1).expand_as(historical)
        attention_input = torch.cat(
            (historical, expanded, historical - expanded, historical * expanded), dim=-1
        )
        logits = self.attention(attention_input).squeeze(-1)
        mask = history_mask.bool()
        logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(logits, dim=1) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        interest = (weights.unsqueeze(-1) * historical).sum(dim=1)
        base = self.base(candidate_x.long()).flatten(start_dim=1)
        return self.scorer(torch.cat((base, interest, candidate), dim=1)).squeeze(-1)


class MetadataSASRec(nn.Module):
    """Small causal Transformer over item/author/tag/duration history."""

    def __init__(
        self, field_dims: Sequence[int], metadata_dims: Mapping[str, int],
        embedding_dim: int = 16, hidden_dim: int = 64, max_seq_len: int = 20,
        num_heads: int = 2, num_layers: int = 1, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.field_dims = [int(value) for value in field_dims]
        self.max_seq_len = int(max_seq_len)
        self.metadata = MetadataEmbedding(metadata_dims, embedding_dim, hidden_dim)
        self.position = nn.Embedding(max_seq_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim * 2, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.base = FeatureEmbedding(self.field_dims, embedding_dim)
        self.scorer = MLP(
            len(self.field_dims) * embedding_dim + hidden_dim * 4,
            (128, 64), dropout=dropout, output_dim=1,
        )

    def forward(
        self, candidate_x: torch.Tensor, history: Mapping[str, torch.Tensor],
        history_mask: torch.Tensor, candidate_metadata: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        token = self.metadata(history)
        batch, length, _ = token.shape
        positions = torch.arange(length, device=token.device).unsqueeze(0).expand(batch, -1)
        token = token + self.position(positions)
        real_mask = history_mask.bool()
        safe_mask = real_mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
        causal = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=token.device), diagonal=1
        )
        encoded = self.encoder(token, mask=causal, src_key_padding_mask=~safe_mask)
        last = safe_mask.long().sum(dim=1).clamp_min(1) - 1
        state = encoded[torch.arange(batch, device=token.device), last]
        if empty.any():
            state = state.clone()
            state[empty] = 0
        candidate = self.metadata(candidate_metadata)
        base = self.base(candidate_x.long()).flatten(start_dim=1)
        interactions = torch.cat((state, candidate, state - candidate, state * candidate), dim=1)
        return self.scorer(torch.cat((base, interactions), dim=1)).squeeze(-1)
