from src.data.tokenizer import Tokenizer
from src.model.assembler import Model
from src.model.definitions import *

NUM_A16_HEADS = 2
FFN_HIDDEN_FEATURES = 3072


def build_model(tokenizer: Tokenizer):
    return Model(
        tokenizer=tokenizer,
        input=Embedding(output_features=384, init_std=0.02),
        blocks=[
            Repeat(
                times=3,
                blocks=[
                    RMSNorm(input_features=384, eps=1e-6),
                    ParallelA16(
                        input_features=384,
                        num_heads=NUM_A16_HEADS,
                        residual=True,
                        rope_base=10000.0,
                    ),
                    SwiGLU(
                        input_features=384,
                        hidden_features=FFN_HIDDEN_FEATURES,
                        output_features=384,
                        residual=True,
                    ),
                    RMSNorm(input_features=384, eps=1e-6),
                ],
            )
        ],
        lm_head=LMHead(input_features=384, tie_to_embedding=True),
    ).assemble()
