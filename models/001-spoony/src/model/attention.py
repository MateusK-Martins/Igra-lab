from torch import Tensor, nn

from src.model.cache_storage import CacheStorage


def _positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class GQAAttentionImpl(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_features: int,
        window_size: int,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_features = head_features
        self.window_size = window_size
        self.dropout = dropout
        self.bias = bias
        self.layer_id = ""

        self.q_proj = nn.Linear(
            input_features,
            num_query_heads * head_features,
            bias=bias,
        )

        self.k_proj = nn.Linear(
            input_features,
            num_kv_heads * head_features,
            bias=bias,
        )

        self.v_proj = nn.Linear(
            input_features,
            num_kv_heads * head_features,
            bias=bias,
        )

        self.out_proj = nn.Linear(
            num_query_heads * head_features,
            output_features,
            bias=bias,
        )

    def assign_id(self, layer_id: int) -> None:
        self.layer_id = f"GQAAttentionImpl-{layer_id}"

    def forward(self, x: Tensor, *, cache: CacheStorage | None = None) -> Tensor:
        batch_size, sequence_length, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(
            batch_size,
            sequence_length,
            self.num_query_heads,
            self.head_features,
        ).transpose(1, 2)

        k = k.reshape(
            batch_size,
            sequence_length,
            self.num_kv_heads,
            self.head_features,
        ).transpose(1, 2)

        v = v.reshape(
            batch_size,
            sequence_length,
            self.num_kv_heads,
            self.head_features,
        ).transpose(1, 2)
