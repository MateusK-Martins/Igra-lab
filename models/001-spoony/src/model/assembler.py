from dataclasses import dataclass

from torch import nn

from src.data.tokenizer import Tokenizer
from src.model.cache_storage import CacheEntry
from src.model.composition import build_block
from src.model.definitions import (
    BlockDefinition,
    EmbeddingDefinition,
    LMHeadDefinition,
)
from src.model.language_model import ModelImpl


@dataclass(frozen=True)
class Model:
    tokenizer: Tokenizer
    input: EmbeddingDefinition
    blocks: list[BlockDefinition]
    lm_head: LMHeadDefinition

    def __post_init__(self) -> None:
        self.validate()

    def _unpack_blocks(self) -> list[BlockDefinition]:
        return [x for block in self.blocks for x in block.unpack()]

    def validate(self) -> None:
        """Check declared values and connections without constructing modules."""
        _require_positive_integer(
            self.tokenizer.vocab_size,
            "tokenizer.vocab_size",
        )
        _require_positive_integer(
            self.input.output_features,
            "input.output_features",
        )
        _require_positive_integer(
            self.lm_head.input_features,
            "lm_head.input_features",
        )

        if type(self.lm_head.tie_to_embedding) is not bool:
            raise ValueError("lm_head.tie_to_embedding must be a boolean")

        current_features = self.input.output_features
        blocks = self._unpack_blocks()

        for index, block in enumerate(blocks):
            block_path = f"blocks[{index}]"

            _require_positive_integer(
                block.input_features,
                f"{block_path}.input_features",
            )
            _require_positive_integer(
                block.output_features,
                f"{block_path}.output_features",
            )

            if type(block.residual) is not bool:
                raise ValueError(f"{block_path}.residual must be a boolean")

            if block.input_features != current_features:
                raise ValueError(
                    f"{block_path} expects "
                    f"{block.input_features} input features, "
                    f"but the previous component outputs "
                    f"{current_features}"
                )

            if block.residual and block.input_features != block.output_features:
                raise ValueError(
                    f"{block_path} residual requires equal input and output features"
                )

            current_features = block.output_features

        if self.lm_head.input_features != current_features:
            raise ValueError("LM head input width does not match the body output")

        if (
            self.lm_head.tie_to_embedding
            and self.lm_head.input_features != self.input.output_features
        ):
            raise ValueError("Tied LM head input width must match embedding width")

    def assemble(self) -> tuple[ModelImpl, list[CacheEntry]]:
        self.validate()

        vocab_size = self.tokenizer.vocab_size
        embedding = self.input.build(vocab_size=vocab_size)

        cache_entries: list[CacheEntry] = []
        blocks = self._unpack_blocks()

        body = nn.ModuleList()

        for block_id, block in enumerate(blocks):
            cache = block.cache(block_id)

            if cache is not None:
                cache_entries.append(cache)

            block_module = build_block(block, block_id)

            body.append(block_module)

        lm_head = self.lm_head.build(
            vocab_size=vocab_size,
            embedding=embedding,
        )

        return (
            ModelImpl(
                embedding=embedding,
                body=body,
                lm_head=lm_head,
            ),
            cache_entries,
        )


def _require_positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
