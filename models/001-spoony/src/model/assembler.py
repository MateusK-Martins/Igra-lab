from dataclasses import dataclass

from torch import nn

from src.data.tokenizer import Tokenizer
from src.model.composition import build_block
from src.model.definitions import (
    BlockDefinition,
    EmbeddingDefinition,
    LMHeadDefinition,
    Repeat,
    Sequential,
)
from src.model.language_model import LanguageModel


@dataclass(frozen=True)
class Model:
    tokenizer: Tokenizer
    input: EmbeddingDefinition
    blocks: list[BlockDefinition]
    lm_head: LMHeadDefinition

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check declared values and connections without constructing modules."""
        _require_positive_integer(self.tokenizer.vocab_size, "tokenizer.vocab_size")
        _require_positive_integer(self.input.output_features, "input.output_features")
        _require_positive_integer(self.lm_head.input_features, "lm_head.input_features")
        if type(self.lm_head.tie_to_embedding) is not bool:
            raise ValueError("lm_head.tie_to_embedding must be a boolean")
        current_features = self.input.output_features

        current_features = self.validate_blocks(current_features, self.blocks)

        if self.lm_head.input_features != current_features:
            raise ValueError("LM head input width does not match the body output")

        if (
            self.lm_head.tie_to_embedding
            and self.lm_head.input_features != self.input.output_features
        ):
            raise ValueError("Tied LM head input width must match embedding width")

    def validate_blocks(
        self,
        current_features: int,
        blocks: list[BlockDefinition],
        *,
        path: str = "blocks",
    ) -> int:
        for index, block in enumerate(blocks):
            block_path = f"{path}[{index}]"
            if isinstance(block, Repeat):
                _require_positive_integer(block.times, f"{block_path}.times")
            if isinstance(block, (Sequential, Repeat)):
                if not block.blocks:
                    raise ValueError(f"{block_path}.blocks must not be empty")
                # Recurse before reading derived properties: a nested repeat
                # may itself have empty children.
                repeated_output = self.validate_blocks(
                    current_features,
                    block.blocks,
                    path=f"{block_path}.blocks",
                )
                if (
                    isinstance(block, Repeat)
                    and block.times > 1
                    and repeated_output != current_features
                ):
                    raise ValueError(
                        f"{block_path} repeat output must match its input when times > 1"
                    )
            _require_positive_integer(
                block.input_features, f"{block_path}.input_features"
            )
            _require_positive_integer(
                block.output_features, f"{block_path}.output_features"
            )
            if type(block.residual) is not bool:
                raise ValueError(f"{block_path}.residual must be a boolean")
            if block.input_features != current_features:
                raise ValueError(
                    f"{block_path} expects {block.input_features} input features, "
                    f"but the previous component outputs {current_features}"
                )

            if block.residual and block.input_features != block.output_features:
                raise ValueError(
                    f"{block_path} residual requires equal input and output features"
                )

            current_features = block.output_features

        return current_features


def _require_positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def assemble(definition: Model) -> LanguageModel:
    definition.validate()
    vocab_size = definition.tokenizer.vocab_size

    embedding = definition.input.build(vocab_size=vocab_size)
    body = nn.Sequential(*(build_block(block) for block in definition.blocks))

    lm_head = definition.lm_head.build(vocab_size=vocab_size, embedding=embedding)

    return LanguageModel(embedding=embedding, body=body, lm_head=lm_head)
