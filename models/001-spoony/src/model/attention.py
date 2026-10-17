import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.model.cache_storage import CacheStorage
from src.model.position import PositionRotation


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
        position_rotation: PositionRotation,
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
        self.position_rotation = position_rotation
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

        start_position = 0 if cache is None else cache.step(self.layer_id)

        positions = torch.arange(
            start_position,
            start_position + sequence_length,
            device=x.device,
        )

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

        q = self.position_rotation(q, positions)
        k = self.position_rotation(k, positions)

        if cache is not None:
            cache.write(self.layer_id, k, v)
            entry = cache.read_all(self.layer_id)

            k = entry.keys
            v = entry.values

            key_start_position = entry.count - k.shape[2]
        else:
            key_start_position = start_position

        key_positions = torch.arange(
            key_start_position,
            key_start_position + k.shape[2],
            device=q.device,
        )

        query_positions = positions[:, None]
        retained_positions = key_positions[None, :]

        attention_mask = retained_positions <= query_positions

        if self.window_size > 0:
            attention_mask &= retained_positions > query_positions - self.window_size

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            enable_gqa=True,
        )

        attended = attended.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            self.num_query_heads * self.head_features,
        )

        return self.out_proj(attended)
