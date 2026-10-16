from src.data.tokenizer import Tokenizer
from src.model.assembler import Model
from src.model.definitions import ACSwiGLU, Embedding, LMHead, Repeat, RMSNorm

MODE = "a16"
CONDITIONED = True
GMLP_HIDDEN_FEATURES = 1024
GMLP_WINDOW_SIZE = 10


def build_model(tokenizer: Tokenizer):
    return Model(
        tokenizer=tokenizer,
        input=Embedding(output_features=384, init_std=0.02),
        blocks=[
            Repeat(
                times=6,
                blocks=[
                    ACSwiGLU(
                        input_features=384,
                        hidden_features=1024,
                        eps=1e-6,
                        rope_base=10000.0,
                        alpha=0.1,
                        conditioned=CONDITIONED,
                        gmlp_hidden_features=GMLP_HIDDEN_FEATURES,
                        gmlp_window_size=GMLP_WINDOW_SIZE,
                    ),
                    RMSNorm(input_features=384, eps=1e-6),
                ],
            )
        ],
        lm_head=LMHead(input_features=384, tie_to_embedding=True),
    ).assemble()
