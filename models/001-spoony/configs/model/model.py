"""Two Tide layers per compressed A16 RoPE layer; preserve the Tide30M FFN and width."""

from functools import partial

from src.data.tokenizer import Tokenizer
from src.model.assembler import Model
from src.model.definitions import (
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
    for layer in range(9):
        mixer = (
            GQAAttention(
                input_features=512,
                output_features=512,
                num_query_heads=1,
                num_kv_heads=1,
                head_features=16,
                window_size=0,
                dropout=0,
                position_rotation=partial(rope, base=10000.0),
                residual=True,
            )
            if layer % 3 == 2
            else NoKVariant(
                input_features=512,
                eps=1e-6,
                residual=True,
                decay_init=0.99,
                mode="write_gate",
                rank=32,
                num_heads=4,
                normalization_scope="global",
                retention_initials=tuple(0.90 + 0.095 * j / 511 for j in range(512)),
                write_initials=tuple(0.20 - 0.18 * j / 511 for j in range(512)),
            )
        )
        blocks.extend(
            [
                RMSNorm(input_features=512, eps=1e-6),
                mixer,
                SwiGLU(
                    input_features=512,
                    hidden_features=1152,
                    output_features=512,
                    residual=True,
                ),
                RMSNorm(input_features=512, eps=1e-6),
            ]
        )
    return Model(
        tokenizer=tokenizer,
        input=Embedding(output_features=512, init_std=0.02),
        blocks=blocks,
        lm_head=LMHead(input_features=512, tie_to_embedding=True),
    ).assemble()
