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
                    CumulativeMixer(
                        input_features=512,
                        eps=1e-6,
                        residual=True,
                        normalization="rms",
                        key_decay=False,
                        value_decay=True,
                        decay_init=0.99,
                        decay_mode="token",
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
