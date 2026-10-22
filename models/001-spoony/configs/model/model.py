"""Llama-style decoder near 30M parameters with a 32K vocabulary."""

from functools import partial

from src.data.tokenizer import Tokenizer
from src.model.assembler import Model
from src.model.definitions import (
    Embedding,
    GQAAttention,
    LMHead,
    PreNorm,
    Repeat,
    RMSNorm,
    SwiGLU,
)
from src.model.position import rope

WIDTH = 512
LAYERS = 5
FFN_WIDTH = 1280
VOCAB_SIZE = 32768


def build_model(tokenizer: Tokenizer):
    return Model(
        tokenizer=tokenizer,
        input=Embedding(output_features=WIDTH, init_std=0.02),
        blocks=[
            Repeat(
                times=LAYERS,
                blocks=[
                    PreNorm(
                        GQAAttention(
                            input_features=WIDTH,
                            output_features=WIDTH,
                            num_query_heads=8,
                            num_kv_heads=2,
                            head_features=64,
                            window_size=0,
                            dropout=0,
                            position_rotation=partial(rope, base=10000.0),
                            bias=False,
                        )
                    ),
                    PreNorm(
                        SwiGLU(
                            input_features=WIDTH,
                            hidden_features=FFN_WIDTH,
                            output_features=WIDTH,
                            bias=False,
                        )
                    ),
                ],
            ),
            RMSNorm(input_features=WIDTH, eps=1e-6),
        ],
        lm_head=LMHead(input_features=WIDTH, tie_to_embedding=True),
    ).assemble()
