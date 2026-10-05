from functools import partial

from src.data.tokenizer import Tokenizer
from src.model.assembler import Model
from src.model.definitions import *
from src.model.position import *


def build_model(tokenizer: Tokenizer):
    return Model(
        tokenizer=tokenizer,
        input=Embedding(output_features=256),
        blocks=[
            Repeat(
                times=4,
                blocks=[
                    GQAAttention(
                        input_features=256,
                        output_features=256,
                        num_query_heads=8,
                        num_kv_heads=2,
                        head_features=32,
                        window_size=128,
                        dropout=0,
                        position_rotation=partial(rope, base=10000.0),
                        residual=True,
                    ),
                    RMSNorm(input_features=256, eps=1e-6),
                    SwiGLU(
                        input_features=256,
                        hidden_features=688,
                        output_features=256,
                        residual=True,
                    ),
                    RMSNorm(input_features=256, eps=1e-6),
                ],
            )
        ],
        lm_head=LMHead(input_features=256, tie_to_embedding=True),
    ).assemble()
