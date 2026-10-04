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

This runnable composition example exercises normalization and feed-forward
blocks. It is not a Transformer: attention computation is still unfinished.
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
SwiGLUImpl accept and ignore it. GQAAttentionImpl accepts it, but does not yet
read or write it because its attention calculation is unfinished.

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

## Verification and remaining work

The full suite passed 200 tests on 2026-10-04. Tests cover normalization and its
gradients/dtypes, expanded order, parameter independence, leaf residuals, head
tying, model save/load, cache IDs, storage writes/snapshots, and forwarding.
The forwarding test substitutes attention's forward; it does not verify real
attention output or causal masking.

GQA currently has Q/K/V/output projections and head reshaping. It still needs
positions, RoPE, history reads, causal/window masks, SDPA, output reshaping and
projection, and writes of new K/V. window_size=0 means unrestricted causal
history; positive values will bound visibility, independently of storage capacity.
Read old history before overwriting circular buffers.

CacheStorage.write() requires matching K/V dtypes before modifying an entry.
A regression test checks that rejection leaves empty buffers/count/pointer
unchanged and that a subsequent valid write can be saved and restored.
