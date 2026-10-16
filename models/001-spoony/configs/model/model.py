"""Tide, fixed-window gMLP, then compressed A16 attention; preserve the Tide14M FFN and width."""

from functools import partial

from src.data.tokenizer import Tokenizer
from src.model.assembler import Model
from src.model.definitions import (
    GMLP,
    Embedding,
    GQAAttention,
    LMHead,
    NoKVariant,
    RMSNorm,
    SwiGLU,
)
from src.model.position import rope


def build_model(tokenizer: Tokenizer):
    blocks = []
    for layer in range(6):
        mixer = (
            GQAAttention(
                input_features=384,
                output_features=384,
                num_query_heads=1,
                num_kv_heads=1,
                head_features=16,
                window_size=0,
                dropout=0,
                position_rotation=partial(rope, base=10000.0),
                residual=True,
            )
            if layer % 3 == 2
            else GMLP(
                input_features=384, hidden_features=1024, window_size=64, residual=True
            )
            if layer % 3 == 1
            else NoKVariant(
                input_features=384,
                eps=1e-6,
                residual=True,
                decay_init=0.99,
                mode="write_gate",
                rank=32,
                num_heads=4,
                normalization_scope="global",
                retention_initials=tuple(0.90 + 0.095 * j / 383 for j in range(384)),
                write_initials=tuple(0.20 - 0.18 * j / 383 for j in range(384)),
            )
        )
        blocks.extend(
            [
                RMSNorm(input_features=384, eps=1e-6),
                mixer,
                SwiGLU(
                    input_features=384,
                    hidden_features=1024,
                    output_features=384,
                    residual=True,
                ),
                RMSNorm(input_features=384, eps=1e-6),
            ]
        )
    return Model(
        tokenizer=tokenizer,
        input=Embedding(output_features=384, init_std=0.02),
        blocks=blocks,
        lm_head=LMHead(input_features=384, tie_to_embedding=True),
    ).assemble()
