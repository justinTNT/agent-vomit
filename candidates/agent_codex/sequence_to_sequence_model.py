from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from .attention_decoder import AttentionDecoder
from .sequence_encoder import SequenceEncoder
from .utils import warn_on_unused_kwargs


class SequenceToSequenceModel(nn.Module):
    """Encoder-decoder architecture for sequence transduction."""

    def __init__(
        self,
        encoder: Optional[SequenceEncoder] = None,
        decoder: Optional[AttentionDecoder] = None,
        target_vocab_size: Optional[int] = None,
        pad_idx: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        warn_on_unused_kwargs(kwargs)

        self.encoder = encoder if encoder is not None else SequenceEncoder()
        self.decoder = decoder if decoder is not None else AttentionDecoder()

        hidden_dim = self.encoder.embedding.embedding_dim
        if self.decoder.d_model != hidden_dim:
            raise ValueError(
                "Encoder and decoder hidden dimensions must match"
            )

        tgt_vocab = target_vocab_size or self.encoder.embedding.num_embeddings
        self.target_embedding = nn.Embedding(
            num_embeddings=tgt_vocab,
            embedding_dim=hidden_dim,
            padding_idx=pad_idx,
        )

        if self.encoder.embedding.embedding_dim != self.decoder.d_model:
            raise ValueError(
                "Encoder and decoder hidden dimensions must match"
            )

    def forward(
        self,
        src_tokens: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoder_outputs = self.encoder(src_tokens, attention_mask=src_mask)
        memory = encoder_outputs["encoded"]
        memory_mask = encoder_outputs["mask"]

        target_embeddings = self.target_embedding(tgt_tokens)

        decoded = self.decoder(
            target_embeddings,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
        )

        return {
            "decoder_output": decoded,
            "encoder_outputs": encoder_outputs,
        }
