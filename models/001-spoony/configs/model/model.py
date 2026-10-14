from functools import partial

from src.data.tokenizer import Tokenizer
from src.model.assembler import Model
from src.model.definitions import *
from src.model.position import *


def build_model(tokenizer: Tokenizer):
    return Model(
        tokenizer=tokenizer,
        input=Embedding(output_features=512, init_std=0.02),
        blocks=[
            Repeat(
                times=9,
                blocks=[
                    RMSNorm(input_features=512, eps=1e-6),
                    GQAAttention(
                        input_features=512,
                        output_features=512,
                        num_query_heads=8,
                        num_kv_heads=2,
                        head_features=64,
                        window_size=512,
                        dropout=0,
                        position_rotation=partial(rope, base=10000.0),
                        residual=True,
                    ),
                    SwiGLU(
                        input_features=512,
                        hidden_features=1408,
                        output_features=512,
                        residual=True,
                    ),
                    RMSNorm(input_features=512, eps=1e-6),
                ],
            )
        ],
        lm_head=LMHead(input_features=512, tie_to_embedding=True),
    ).assemble()
