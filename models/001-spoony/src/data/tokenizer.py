"""Train, save, and use a byte-level BPE tokenizer."""

from collections.abc import Iterable, Sequence
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer


class ByteLevelBPETokenizer:
    """A reversible BPE tokenizer built from UTF-8 byte-level pieces."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        """Return the actual vocabulary size learned during training."""
        return self._tokenizer.get_vocab_size()

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        target_vocab_size: int,
        min_frequency: int,
        special_tokens: Sequence[str],
    ) -> "ByteLevelBPETokenizer":
        """Learn BPE merges from training text only."""
        if target_vocab_size <= 0:
            raise ValueError("target_vocab_size must be greater than zero")
        if min_frequency <= 0:
            raise ValueError("min_frequency must be greater than zero")
        if not special_tokens:
            raise ValueError("special_tokens must include an unknown token")

        tokenizer = Tokenizer(BPE(unk_token=special_tokens[0]))
        tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=target_vocab_size,
            min_frequency=min_frequency,
            special_tokens=list(special_tokens),
            initial_alphabet=ByteLevelPreTokenizer.alphabet(),
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)

        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        """Encode text as BPE token IDs without adding special tokens."""
        return self._tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode BPE token IDs back to text."""
        return self._tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def token_to_id(self, token: str) -> int:
        """Return a known token's ID or fail with a clear error."""
        token_id = self._tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"token {token!r} is not in this vocabulary")
        return token_id

    def save(self, path: Path) -> None:
        """Save the complete tokenizer vocabulary and merge rules as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "ByteLevelBPETokenizer":
        """Load a tokenizer previously saved with :meth:`save`."""
        return cls(Tokenizer.from_file(str(path)))
