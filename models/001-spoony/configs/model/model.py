from src.data.tokenizer import Tokenizer
from src.model.assembler import Model
from src.model.definitions import *

MODE = "a16"
MATRIX_MODE = "single"
RUN_NAME = "a16-diagonal-matrix-14m"

SINGLE_SCALE = (0.99,)
MULTI_SCALE = (0.99, 0.90, 0.99, 0.999)


def build_model(tokenizer: Tokenizer):
    return Model(
        tokenizer=tokenizer,
        input=Embedding(output_features=384, init_std=0.02),
        blocks=[
            Repeat(
                times=6,
                blocks=[
                    RMSNorm(input_features=384, eps=1e-6),
                    LatentAttention(
                        input_features=384, mode=MODE, residual=True, rope_base=10000.0
                    ),
                    RMSNorm(input_features=384, eps=1e-6),
                    MatrixFastWeights(
                        input_features=384,
                        timescales=SINGLE_SCALE
                        if MATRIX_MODE == "single"
                        else MULTI_SCALE,
                        learned_decay=True,
                        residual=True,
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
