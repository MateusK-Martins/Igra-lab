# Pair-programming continuation — 2026-10-04

Branch: model/001-spoony. This checkpoint includes RMSNorm, model assembly
refactoring, external cache storage, and cache forwarding. Read current code
before guiding edits; preserve subsequent user changes.

## Current implementation

The data pipeline, tokenizer, disk-backed windows, DataLoaders, embeddings,
SwiGLU, RMSNorm, leaf residuals, and head tying are implemented. No real corpus,
final architecture, training loop, or complete attention is configured.

Model definitions inherit protocols with default cache() and unpack() methods.
Sequential/Repeat unpack recursively into leaf definitions; they no longer build
runtime groups. Their residual fields remain present but are not applied. The
user chose leaf residuals and explicit ModuleList iteration; do not reintroduce
composition wrappers or a cache-aware runtime hierarchy.

Model.validate() checks the expanded sequence. Model.assemble() returns
(ModelImpl, list[CacheEntry]), builds independent modules per occurrence, and
uses expanded integer indexes for identity. build_block assigns IDs before
wrapping leaf residuals. Runtime names use Impl.

All body runtimes accept forward(x, *, cache=None). ModelImpl explicitly loops
over self.body, forwarding the same external CacheStorage; ResidualImpl passes
it to its branch. LinearImpl, RMSNormImpl, and SwiGLUImpl ignore cache. Embedding
and LM head remain standard modules. No storage is retained in the model.

GQAAttention.cache(id) and runtime assign_id(id) use matching
GQAAttentionImpl-{id} keys. Definitions produce independent unbounded empty K/V
entries [0, kv_heads, 0, head_features]. First write establishes batch and dtype.
Batch size/capacity do not belong in the assembly API.

## Storage contract

CacheEntry has layer_id, bounded, keys, values, count, pointer. K/V layout is
[batch, kv_heads, tokens, head_features]. count is total tokens written.
Bounded entries use preallocated circular buffers; pointer is the next write
slot. Bulk writes copy at most two spans and retain the final capacity tokens
for oversized chunks. Unlimited entries append all tokens.

CacheStorage(entries, storage_device, read_device) supports step(), write(),
chronological read_all(), save(), and load(). read_all's pointer=0 describes a
read result, not physical state. Some reads share buffers; treat them as read-only.
CPU RAM and CUDA VRAM are selectable; disk is snapshot persistence only.
Snapshots contain plain dictionaries and detached CPU tensors. load() validates
physical state with weights_only=True and accepts caller-selected devices.

## Next work and known issues

GQAAttentionImpl.forward receives cache, projects Q/K/V, and reshapes heads, but
still has no output. Implement positions/RoPE, cache history concatenation,
causal/window masks, SDPA, output reshaping/projection, and new-K/V writes.
Read history before overwriting bounded buffers. Use GQA from the outset.
window_size=0 means unrestricted causal history; positive values bound attention
visibility independently of retained storage capacity.

CacheStorage.write() rejects mismatched K/V dtypes before buffer initialization
or writes. Regression coverage checks unchanged state after rejection, then a
valid write and snapshot restoration.

Flattening changed body state-dictionary paths. Same-definition round trips are
tested; migration of old nested state dictionaries is not implemented.

## Verification

200 tests passed on 2026-10-04. Full Ruff lint and format checks passed.
Coverage includes RMSNorm numerical behavior/gradients, expanded composition
order, independent parameters, residuals, tying, cache identities, circular and
unlimited storage, snapshot continuation, CPU/CUDA restoration, and forwarding.
Attention forwarding uses a substituted forward: no real causal-attention or
cached-decoding equivalence test exists yet. See MODEL_ASSEMBLY.md for the API.
