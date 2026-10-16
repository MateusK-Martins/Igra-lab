from src.data.tokenizer import Tokenizer
from src.model.assembler import Model
from src.model.definitions import ACSwiGLU, Embedding, LMHead, Repeat, RMSNorm

MODE = "a16"
CONDITIONED = True
MODEL_WIDTH = 512
LAYERS = 14
FFN_HIDDEN = 1152


def build_model(tokenizer: Tokenizer):
    return Model(
        tokenizer=tokenizer,
        input=Embedding(output_features=MODEL_WIDTH, init_std=0.02),
        blocks=[
            Repeat(
                times=LAYERS,
                blocks=[
                    ACSwiGLU(
                        input_features=MODEL_WIDTH,
                        hidden_features=FFN_HIDDEN,
                        eps=1e-6,
                        rope_base=10000.0,
                        alpha=0.1,
                        conditioned=CONDITIONED,
                    ),
                    RMSNorm(input_features=MODEL_WIDTH, eps=1e-6),
                ],
            )
        ],
        lm_head=LMHead(input_features=MODEL_WIDTH, tie_to_embedding=True),
    ).assemble()
