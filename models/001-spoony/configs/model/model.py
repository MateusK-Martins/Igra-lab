"""70M T16: six Tide mixers, three A16 attention layers, FFN width 2368."""

from functools import partial

from src.data.tokenizer import Tokenizer
from src.model.assembler import Model
from src.model.definitions import (
    Embedding,
    GQAAttention,
    LMHead,
    RMSNorm,
    SwiGLU,
    TideMixer,
)
from src.model.position import rope


def build_model(tokenizer: Tokenizer):
    blocks = []
    for layer in range(9):
        mixer = (
            GQAAttention(
                input_features=768,
                output_features=768,
                num_query_heads=1,
                num_kv_heads=1,
                head_features=16,
                window_size=0,
                dropout=0,
                position_rotation=partial(rope, base=10000.0),
                residual=True,
            )
            if layer % 3 == 2
            else TideMixer(
                input_features=768,
                eps=1e-6,
                residual=True,
                retention_initials=tuple(0.90 + 0.095 * j / 767 for j in range(768)),
                write_initials=tuple(0.20 - 0.18 * j / 767 for j in range(768)),
            )
        )
        blocks.extend(
            [
                RMSNorm(input_features=768, eps=1e-6),
                mixer,
                SwiGLU(
                    input_features=768,
                    hidden_features=2368,
                    output_features=768,
                    residual=True,
                ),
                RMSNorm(input_features=768, eps=1e-6),
            ]
        )
    return Model(
        tokenizer=tokenizer,
        input=Embedding(output_features=768, init_std=0.016329931618554522),
        blocks=blocks,
        lm_head=LMHead(input_features=768, tie_to_embedding=True),
    ).assemble()
