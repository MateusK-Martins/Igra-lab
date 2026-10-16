"""Four pre-norm attention -> TDM -> SwiGLU layers, approximately14M parameters."""

from functools import partial

from src.data.tokenizer import Tokenizer
from src.model.assembler import Model
from src.model.definitions import (
    Embedding,
    GQAAttention,
    LMHead,
    RMSNorm,
    SwiGLU,
    TemporalDifferenceMixer,
)
from src.model.position import rope


def build_model(tokenizer: Tokenizer):
    blocks = []
    for _ in range(4):
        blocks.extend(
            [
                RMSNorm(input_features=384, eps=1e-6),
                GQAAttention(
                    input_features=384,
                    output_features=384,
                    num_query_heads=8,
                    num_kv_heads=2,
                    head_features=48,
                    window_size=256,
                    dropout=0,
                    position_rotation=partial(rope, base=10000.0),
                    residual=True,
                ),
                RMSNorm(input_features=384, eps=1e-6),
                TemporalDifferenceMixer(
                    input_features=384, scales=(1, 2, 4, 8, 16, 32), residual=True
                ),
                RMSNorm(input_features=384, eps=1e-6),
                SwiGLU(
                    input_features=384,
                    hidden_features=1024,
                    output_features=384,
                    residual=True,
                ),
            ]
        )
    blocks.append(RMSNorm(input_features=384, eps=1e-6))
    return Model(
        tokenizer=tokenizer,
        input=Embedding(output_features=384, init_std=0.02),
        blocks=blocks,
        lm_head=LMHead(input_features=384, tie_to_embedding=True),
    ).assemble()
