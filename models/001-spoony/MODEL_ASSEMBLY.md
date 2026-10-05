# Declarative model assembly

Model settings are Python dataclass definitions. Editable architecture settings
will live in `configs/model/architecture.py`; no final architecture is configured
yet. Importing settings must not download data or train a tokenizer.

## Current API

Supply an already trained tokenizer when constructing the definition:

```python
definition = Model(
    tokenizer=tokenizer,
    input=Embedding(output_features=256),
    blocks=[
        Repeat(
            times=3,
            blocks=[
                RMSNorm(input_features=256, eps=1e-6),
                SwiGLU(
                    input_features=256,
                    hidden_features=688,
                    output_features=256,
                    residual=True,
                ),
            ],
        ),
    ],
    lm_head=LMHead(input_features=256, tie_to_embedding=True),
)
model, cache_entries = definition.assemble()
logits = model(token_ids)
```

This composition example exercises normalization and feed-forward blocks.
GQA is implemented, but the final Transformer architecture is not configured.
The former module-level `assemble(definition)` function has been replaced by
`Model.assemble()`.

Definitions use plain names. Custom runtime modules use `Impl`: ModelImpl,
ResidualImpl, LinearImpl, SwiGLUImpl, RMSNormImpl, and GQAAttentionImpl.
Embedding and LM head use standard PyTorch modules.

## Definition contract

`BlockDefinition` exposes `input_features`, `output_features`, `residual`, and:

- `build()` returns a fresh, unwrapped runtime module.
- `cache(layer_id)` returns a CacheEntry or None.
- `unpack()` returns the leaf definitions in execution order.

Concrete definitions inherit the protocol to reuse its default `cache()`
(returning None) and `unpack()` (returning `[self]`). A structurally compatible
custom definition must supply these methods itself if it does not inherit them.
Its runtime module must accept `forward(x, *, cache=None)`.

Feature dimensions and repeat counts must be positive integers; booleans are
rejected. Leaves validate their component settings at construction.

## Expansion and validation

Sequential recursively concatenates its children's unpacked definitions.
Repeat does the same for each repetition:

```text
[A, Repeat(2, [B, Sequential([C, D])]), E]
    -> [A, B, C, D, B, C, D, E]
```

Expansion reuses definition objects, but assembly builds fresh runtime modules
and parameters for every occurrence. Neither composition builds a runtime group;
calling its `build()` raises TypeError. Empty compositions are rejected, including
when a child list is cleared after construction and the model is revalidated.

Model.validate() walks the expanded sequence and checks positive widths,
boolean residuals, neighboring widths, residual compatibility, and connections
to embedding and LM head. Errors name expanded indexes such as `blocks[3]`.
Repeat-boundary compatibility follows from those neighboring-width checks.
Constructor validation runs immediately; assembly revalidates before building.
An empty model body may connect embedding directly to the head.

Sequential and Repeat serve as declaration groups only. Their existing
`residual` fields are not applied by expansion; use residuals on leaf blocks.
The former residuals around whole composition groups are no longer supported.

## Construction and execution

Model.assemble() enumerates the expanded leaves, collects `cache(block_id)`,
and calls `build_block(block, block_id)`. build_block assigns IDs to modules
implementing `assign_id(int)` before applying the leaf's residual wrapper.
The constructed modules are registered in a flat nn.ModuleList.

The runtime model embeds token IDs, explicitly iterates through that list, and
projects the final hidden states to vocabulary logits:

```text
IDs [B, T] -> embedding [B, T, D] -> body -> logits [B, T, V]
```

Each ordinary block transforms the last dimension while preserving batch and
sequence dimensions. ResidualImpl computes `x + branch(x, cache=cache)` and
rejects unequal input/output shapes. No automatic projection is inserted.
Input-rank and context-limit checks are not implemented yet.

The tokenizer supplies the actual vocabulary size. A bias-free tied head shares
the embedding's Parameter object and requires matching widths. Rebuilding from
the same definition before loading a state dictionary restores that relationship.
Flattening changes body parameter paths compared with the previous nested
runtime: old state dictionaries require an explicit migration.

## External cache

```python
storage = CacheStorage(
    cache_entries,
    storage_device=torch.device("cpu"),
    read_device=torch.device("cuda"),
)
logits = model(token_ids, cache=storage)
```

The model does not retain storage. It passes the same optional instance to every
body block; residuals forward it to their branch. LinearImpl, RMSNormImpl, and
SwiGLUImpl accept and ignore it. GQAAttentionImpl computes positions from the
pre-write count, rotates new Q/K, writes new K/V, and then reads retained history.
It derives absolute key positions from the post-write count and retained length.

GQA IDs are `GQAAttentionImpl-{expanded_index}`. Each occurrence creates its own
entry with K/V shape `[0, kv_heads, 0, head_features]`, count=0, pointer=0, and
bounded=False. First write establishes batch size and dtype. Assembly supplies
no batch size or storage capacity. Bounded storage can instead be constructed
with explicit preallocated entries; attention visibility is a separate setting.

CacheStorage supports unlimited append or circular bounded writes, chronological
reads, device transfers, and plain-dictionary snapshots. count tracks all tokens
written; pointer is the next physical write slot. read_all returns a read result
with pointer=0, transferring tensors to read_device. Returned tensors can share
storage and should be treated as read-only.

save() writes detached CPU tensors and physical state. load() uses
weights_only=True, validates entries, and restores on caller-selected devices.
Snapshots are disk persistence, not a live disk cache.

## Positional rotation and attention

GQAAttention requires a PositionRotation callable before the default bias and
residual fields. Bind RoPE's base explicitly in architecture settings:

```python
from functools import partial
from src.model.position import rope

rotation = partial(rope, base=10000.0)
attention = GQAAttention(
    input_features=256, output_features=256,
    num_query_heads=8, num_kv_heads=2, head_features=32,
    window_size=128, dropout=0.0, position_rotation=rotation,
    residual=True,
)
```

The callable accepts (x, positions) and preserves [B, H, T, D], device and dtype.
RoPE rotates adjacent pairs and requires positive even D. Its base must be finite
and positive; FP16/BF16 arithmetic is promoted to float32 and cast back. Positions
are supplied by attention, with the cache's total count as the starting offset.
Cached keys are already rotated; only new keys are rotated on each call.

Attention constructs an explicit boolean [query_tokens, retained_tokens] mask.
True permits attention when key_position <= query_position and, for positive
window_size W, key_position > query_position - W. W includes the current token;
zero means unrestricted causal history. SDPA uses enable_gqa=True,
is_causal=False, and zero dropout in evaluation. Heads are merged before out_proj.

Write-before-read is the chosen cache policy. Unlimited caches reproduce full
uncached execution across chunks. Bounded caches reproduce windowed execution
for single-token calls when capacity covers the attention window. Multi-token
writes may overwrite context needed by earlier queries; oversized chunks can
leave earlier queries with no permitted keys. Tests intentionally cover this
post-write retention behavior. Fully masked queries produce zero attention
branches on the tested backend; out_proj bias and residuals can still contribute.
No automatic chunking or history preservation is implemented.

## Verification and remaining work

The full suite passed 260 tests on 2026-10-04, including CUDA checks on the local
machine. Reference tests independently calculate GQA scores and gradients.
Coverage also includes RoPE scalar rotations, norms, relative-position behavior,
chunk offsets/dtypes/gradients, future-token isolation, exact window boundaries,
evaluation dropout, snapshot continuation, and real repeated-model cached logits.
CUDA attention is checked with CPU storage and CUDA reads.

CacheStorage.write() requires matching K/V dtypes before modifying an entry.
Regression coverage checks rejection without mutation and subsequent valid
write/save/load. Cached tests run under no_grad; bounded training/backpropagation
through mutable cache buffers is not covered. Training normally uses cache=None.

Next work is the editable Transformer architecture, corpus configuration, and
training integration. Input-rank/context-limit validation, migration of old nested
state dictionaries, and optimization of explicit masks remain future work.
