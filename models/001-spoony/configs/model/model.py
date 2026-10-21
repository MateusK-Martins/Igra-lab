"""Lantern attention with nine shared TokenMLPs; 20,723,200 params at V=8192."""

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


def build_model(tokenizer: Tokenizer):
    blocks = []
    for _ in range(9):
        feedforward = TokenMLP(
            input_features=768,
            bottleneck_features=64,
            chunk_size=2048,
            init_std=0.02,
            residual=True,
        )
        blocks.extend(
            [
                RMSNorm(input_features=768, eps=1e-6),
                GQAAttention(
                    input_features=768,
                    output_features=768,
                    num_query_heads=8,
                    num_kv_heads=2,
                    head_features=96,
                    window_size=0,
                    dropout=0,
                    position_rotation=partial(rope, base=10000.0),
                    residual=True,
                ),
                feedforward,
                RMSNorm(input_features=768, eps=1e-6),
            ]
        )
    return Model(
        tokenizer=tokenizer,
        input=Embedding(output_features=768, init_std=0.016329931618554522),
        blocks=blocks,
        lm_head=LMHead(input_features=768, tie_to_embedding=True),
    ).assemble()
