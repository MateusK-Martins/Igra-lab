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
                    NoKVariant(
                        input_features=384,
                        eps=1e-6,
                        residual=True,
                        decay_init=0.99,
                        mode="write_gate",
                        rank=32,
                        num_heads=4,
                        retention_initials=(0.90, 0.95, 0.98, 0.995),
                        write_initials=(0.50, 0.50, 0.50, 0.50),
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
