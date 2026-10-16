# Tide model assembly

Editable definitions live in configs/model/model.py; runtime implementation
lives in src/model/. Definitions expose dimensions, residual and build().
build() returns an unwrapped branch; build_block adds the declared residual once.
Model.assemble() expands Repeat/Sequential into a flat ModuleList and creates
independent modules/cache entries for every occurrence.

Each default layer is RMSNorm -> residual TideMixer -> residual SwiGLU -> RMSNorm.
Embedding and LM head are tied. TideMixer projects full-width Q and signed V,
plus independent feature-wise retention and write gates. There are no keys,
attention heads, matrix memories, alternative normalization/readout modes or routers.

TideMixer(input_features,eps=1e-6,retention_initials=None,write_initials=None,
residual=False) accepts per-feature initial probability tuples. None selects
the continuous Tide ranges. Probabilities must be strictly between0and1.
Gate weights initialize to zero; biases are logits. RMSNorm has learned
per-feature gains and uses all features in the token.

TideMixerImpl uses the existing vectorized affine scan for the signed update.
BF16/FP16 writes, gates and state accumulation use FP32; FP64 references retain
their precision. Output casts back to query dtype. The module has no internal
persistent cache. Optional external CacheStorage stores raw S as value_sum
[B,1,D] and an empty key_sum [B,1,0]. Cached calls require no_grad.
StateEntry IDs are TideMixerImpl-<expanded_index>. Full sequence, chunks,
single tokens and saved/restored cache states produce equivalent outputs.
CacheStorage handles recurrent StateEntry objects only. The empty key_sum field
and format-v1 snapshot schema are retained for recurrent snapshot compatibility;
attention KV snapshots are rejected. There is no ring buffer or growing KV cache.

A temporary initialization tensor preserves the historical RNG draw formerly
consumed by an unused parent projection. It is not a parameter, projection,
runtime state or forward operation. This keeps initial weights and subsequent
layer initialization compatible while removing the experimental parent classes.
