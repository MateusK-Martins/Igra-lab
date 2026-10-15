from src.data.tokenizer import Tokenizer
from src.model.assembler import Model
from src.model.definitions import *


def build_model(tokenizer: Tokenizer):
    return Model(
        tokenizer=tokenizer,
        input=Embedding(output_features=512, init_std=0.02),
        blocks=[
            Repeat(
                times=9,
                blocks=[
                    RMSNorm(input_features=512, eps=1e-6),
                    NoKVariant(
                        input_features=512,
                        eps=1e-6,
                        residual=True,
                        decay_init=0.99,
                        mode="write_gate",
                        rank=32,
                        num_heads=4,
                        normalization_scope="global",
                        retention_initials=tuple(
                            0.90 + 0.095 * j / 511 for j in range(512)
                        ),
                        write_initials=tuple(0.20 - 0.18 * j / 511 for j in range(512)),
                    ),
                    SwiGLU(
                        input_features=512,
                        hidden_features=1152,
                        output_features=512,
                        residual=True,
                    ),
                    RMSNorm(input_features=512, eps=1e-6),
                ],
            )
        ],
        lm_head=LMHead(input_features=512, tie_to_embedding=True),
    ).assemble()
