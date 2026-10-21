"""Lantern attention with six shared TokenMLPs and an RMSNorm context vector."""

from functools import partial

from src.data.tokenizer import Tokenizer
from src.model.assembler import Model
from src.model.definitions import (
    Embedding,
    GQAAttention,
    LMHead,
    RMSNorm,
    TokenMLP,
)
from src.model.position import rope

TOKEN_MLP_CONTEXT_NORM_EPS = 1e-6
TOKEN_MLP_CONTEXT_VECTOR_SCALE = 0.02


def build_model(tokenizer: Tokenizer):
    blocks = []
    for _ in range(6):
        feedforward = TokenMLP(
            input_features=384,
            bottleneck_features=64,
            chunk_size=2048,
            init_std=0.02,
            context_norm_eps=TOKEN_MLP_CONTEXT_NORM_EPS,
            context_vector_scale=TOKEN_MLP_CONTEXT_VECTOR_SCALE,
            residual=True,
        )
        blocks.extend(
            [
                RMSNorm(input_features=384, eps=1e-6),
                GQAAttention(
                    input_features=384,
                    output_features=384,
                    num_query_heads=8,
                    num_kv_heads=2,
                    head_features=48,
                    window_size=256,
                    dropout=0,
                    position_rotation=partial(rope, base=10000.0),
                    residual=True,
                ),
                feedforward,
                RMSNorm(input_features=384, eps=1e-6),
            ]
        )
    return Model(
        tokenizer=tokenizer,
        input=Embedding(output_features=384, init_std=0.02),
        blocks=blocks,
        lm_head=LMHead(input_features=384, tie_to_embedding=True),
    ).assemble()
