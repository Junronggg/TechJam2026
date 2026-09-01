from __future__ import annotations

import torch
from torch import nn


class SASRecCandidateScorer(nn.Module):
    """
    Lightweight SASRec-style candidate scorer.

    This model encodes a user's strictly-past video history with a
    Transformer and scores the current candidate video against the
    resulting user-state representation.

    IMPORTANT
    ---------
    history_video_ids must contain ONLY interactions strictly before
    the current row. Current/future interactions must never appear.

    Padding ID:
        0

    Therefore real video IDs should be encoded from:
        1 ... num_videos

    Inputs
    ------
    history_video_ids:
        LongTensor [batch_size, seq_len]

    history_mask:
        BoolTensor [batch_size, seq_len]

        True  = real historical interaction
        False = padding

    candidate_video_id:
        LongTensor [batch_size]

    Output
    ------
    Raw score [batch_size]

    Do NOT apply sigmoid inside this model so it can later support
    BCE or BPR objectives.
    """

    def __init__(
        self,
        num_videos: int,
        hidden_dim: int = 64,
        max_seq_len: int = 50,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if num_videos <= 0:
            raise ValueError(
                "num_videos must be greater than 0"
            )

        if max_seq_len <= 0:
            raise ValueError(
                "max_seq_len must be greater than 0"
            )

        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_heads"
            )

        self.num_videos = int(num_videos)
        self.hidden_dim = int(hidden_dim)
        self.max_seq_len = int(max_seq_len)

        # +1 because ID 0 is reserved for padding.
        self.video_embedding = nn.Embedding(
            self.num_videos + 1,
            self.hidden_dim,
            padding_idx=0,
        )

        self.position_embedding = nn.Embedding(
            self.max_seq_len,
            self.hidden_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.layer_norm = nn.LayerNorm(
            self.hidden_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        nn.init.normal_(
            self.video_embedding.weight,
            mean=0.0,
            std=0.02,
        )

        # Padding representation should remain zero.
        with torch.no_grad():
            self.video_embedding.weight[0].zero_()

        nn.init.normal_(
            self.position_embedding.weight,
            mean=0.0,
            std=0.02,
        )

    def encode_history(
        self,
        history_video_ids: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode each user's historical sequence into one state vector.
        """

        if history_video_ids.ndim != 2:
            raise ValueError(
                "history_video_ids must have shape "
                "[batch_size, seq_len]"
            )

        if history_mask.shape != history_video_ids.shape:
            raise ValueError(
                "history_mask must have the same shape "
                "as history_video_ids"
            )

        batch_size, seq_len = (
            history_video_ids.shape
        )

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )

        history_video_ids = (
            history_video_ids.long()
        )

        history_mask = (
            history_mask.bool()
        )

        positions = torch.arange(
            seq_len,
            device=history_video_ids.device,
        )

        positions = (
            positions
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        x = (
            self.video_embedding(
                history_video_ids
            )
            + self.position_embedding(
                positions
            )
        )

        x = self.dropout(x)

        # Transformer expects True for positions that should be ignored.
        padding_mask = ~history_mask

        # Causal mask:
        # token at position i cannot see positions > i.
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=history_video_ids.device,
            ),
            diagonal=1,
        )

        encoded = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )

        encoded = self.layer_norm(
            encoded
        )

        # Number of genuine historical items for each sample.
        lengths = (
            history_mask
            .long()
            .sum(dim=1)
        )

        # Some rows may have no history.
        #
        # Clamp so indexing remains valid.
        safe_lengths = lengths.clamp(
            min=1
        )

        last_positions = (
            safe_lengths - 1
        )

        batch_indices = torch.arange(
            batch_size,
            device=encoded.device,
        )

        user_state = encoded[
            batch_indices,
            last_positions,
        ]

        # If there is actually no history, use zero state rather
        # than an arbitrary positional representation.
        no_history = (
            lengths == 0
        )

        if no_history.any():
            user_state = user_state.clone()
            user_state[no_history] = 0.0

        return user_state

    def forward(
        self,
        history_video_ids: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_video_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score the current candidate against the encoded user history.
        """

        if candidate_video_id.ndim != 1:
            raise ValueError(
                "candidate_video_id must have shape [batch_size]"
            )

        if (
            candidate_video_id.shape[0]
            != history_video_ids.shape[0]
        ):
            raise ValueError(
                "candidate batch size does not match history batch size"
            )

        user_state = self.encode_history(
            history_video_ids,
            history_mask,
        )

        candidate_embedding = (
            self.video_embedding(
                candidate_video_id.long()
            )
        )

        # Dot-product candidate scoring.
        score = (
            user_state
            * candidate_embedding
        ).sum(dim=-1)

        # Standard scaling helps stabilize large embedding dimensions.
        score = score / (
            self.hidden_dim ** 0.5
        )

        return score