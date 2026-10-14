from src.data.tokenizer import Tokenizer
from src.model.assembler import Model
from src.model.definitions import *


def build_model(tokenizer: Tokenizer):
    return Model(
        tokenizer=tokenizer,
        input=Embedding(output_features=384, init_std=0.02),
        blocks=[
            Repeat(
                times=6,
                blocks=[
                    RMSNorm(input_features=384, eps=1e-6),
                    CumulativeMixer(
                        input_features=384,
                        eps=1e-6,
                        residual=True,
                        normalization="rms",
                        key_decay=False,
                        value_decay=True,
                        decay_init=0.99,
                        decay_mode="token",
                        normalization_scope="head",
                        readout="matrix",
                        num_heads=6,
                        query_head_features=48,
                        head_mlp_hidden=24,
                    ),
                    SwiGLU(
                        input_features=384,
                        hidden_features=1024,
                        output_features=384,
                        residual=True,
                    ),
                    RMSNorm(input_features=384, eps=1e-6),
                ],
            )
        ],
        lm_head=LMHead(input_features=384, tie_to_embedding=True),
    ).assemble()
