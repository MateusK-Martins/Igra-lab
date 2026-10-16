"""Six A16 layers followed by independent compressed-gMLP residual blocks."""

from functools import partial

from src.data.tokenizer import Tokenizer
from src.model.assembler import Model
from src.model.definitions import (
    CompressedGMLP,
    Embedding,
    GQAAttention,
    LMHead,
    RMSNorm,
    SwiGLU,
)
from src.model.position import rope


def build_model(tokenizer: Tokenizer):
    blocks = []
    for layer in range(6):
        mixer = GQAAttention(
            input_features=384,
            output_features=384,
            num_query_heads=1,
            num_kv_heads=1,
            head_features=16,
            window_size=0,
            dropout=0,
            position_rotation=partial(rope, base=10000.0),
            residual=True,
        )
        blocks.extend(
            [
                RMSNorm(input_features=384, eps=1e-6),
                mixer,
                RMSNorm(input_features=384, eps=1e-6),
                CompressedGMLP(
                    input_features=384,
                    latent_features=16,
                    hidden_features=64,
                    window_size=64,
                    residual=True,
                ),
                SwiGLU(
                    input_features=384,
                    hidden_features=1024,
                    output_features=384,
                    residual=True,
                ),
                RMSNorm(input_features=384, eps=1e-6),
            ]
        )
    return Model(
        tokenizer=tokenizer,
        input=Embedding(output_features=384, init_std=0.02),
        blocks=blocks,
        lm_head=LMHead(input_features=384, tie_to_embedding=True),
    ).assemble()
