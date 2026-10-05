# Pair-programming continuation — 2026-10-04

Branch: model/001-spoony. This checkpoint includes RMSNorm, model assembly
refactoring, external cache storage, cache forwarding, configured architecture,
and initial training components. Read current code
before guiding edits; preserve subsequent user changes.

## Current implementation

The data pipeline, tokenizer, disk-backed windows, DataLoaders, embeddings,
SwiGLU, RMSNorm, leaf residuals, head tying, and attention are implemented. The
small decoder architecture is configured. No real corpus or training loop is
configured.

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

## Attention implementation and next work

GQAAttention requires an injected PositionRotation callable, declared before
fields with defaults. src/model/position.py defines the callable protocol and
rope(x, positions, *, base). Bind base with functools.partial in editable
settings. RoPE uses adjacent pairs and requires even head width. Attention
constructs absolute positions and rotates only new Q/K; V is unchanged.

The user chose write -> read inside attention. Do not replace it with history
concatenation before writing. Total pre-write count sets new token positions;
post-write count minus retained length sets the first key position. An explicit
boolean mask enforces causal and optional window visibility. window_size=0 is
unrestricted causal history; positive windows include the current token.
SDPA uses enable_gqa=True, is_causal=False, and training-only dropout. The result
is merged across query heads and projected to output_features.

Unlimited cached chunks match full execution. Bounded single-token calls match
windowed execution when capacity covers the window. Bounded chunks may lose
context for earlier queries; oversized chunks can fully mask earlier queries.
This is accepted post-write retention behavior, covered explicitly by tests.
No automatic chunk splitting or history preservation is implemented.

CacheStorage.write() rejects mismatched K/V dtypes before mutation. Cached tests
use no_grad; bounded cached backpropagation is not tested. Training uses no cache.
Flattening changed state-dictionary paths; old nested migration remains pending.

## Configured model and training infrastructure

configs/model/model.py exposes build_model(tokenizer), returning (model, entries).
It receives a trained tokenizer; loading/preparation belongs to the calling
script. Architecture: width 256, four layers of residual GQA -> RMSNorm ->
residual SwiGLU -> RMSNorm. GQA uses 8 query heads, 2 KV heads, head width 32,
window 128, dropout 0, partial(rope, base=10000.0). SwiGLU hidden width 688;
RMSNorm epsilon 1e-6; head tied to embedding. Four cache IDs have expanded indexes
0, 4, 8, 12. Vocabulary 4096 gives 3,819,520 unique parameters.

The user chose default PyTorch initialization and torch.manual_seed(42) directly
in scripts/__init__.py. Do not add an initializer or seed abstraction. Importing
scripts sets the PyTorch seed; DataLoaders have separate configured generators.

src/training/loss.py implements mean next-token cross-entropy without another
target shift. optimizer.py constructs AdamW over model.parameters() using
validated OptimizerConfig. All parameters receive the declared weight decay;
tied weights are deduplicated by PyTorch. scheduler.py constructs LambdaLR from
validated SchedulerConfig, with positive first-update warmup LR, cosine decay
to the floor on the final planned update, and a constant floor afterwards.
Without warmup the first update uses base LR, even for total_steps=1.
Editable configs/training values are provisional: LR 3e-4, betas (0.9, 0.95),
epsilon 1e-8, decay 0.01; warmup 100, total 1000, minimum LR ratio 0.1.

Scheduler construction sets the initial LR. Step it after successful optimizer
updates only. Resume rebuilds both objects with the same settings, then loads
both state dictionaries; LambdaLR does not save the closure's configuration.

Next agreed step: src/training/precision.py for FP32/BF16/FP16 execution, then
the training step with accumulation and clipping. FP16 requires GradScaler;
if it skips an optimizer update, the scheduler must not advance. The user wants
infrastructure completed before selecting a corpus or starting training.

## Verification

323 tests passed on 2026-10-04, including CUDA tests. Ruff lint and formatting
checks passed for src, configs, scripts, and tests.
New tests cover explicit GQA output/gradient reference, causal/window isolation,
RoPE scalar reference/norms/offsets/dtypes/gradients, dropout, cached snapshots,
bounded write-first semantics, real repeated-model logits, and CUDA attention
with CPU storage. Existing identity/forwarding/assembly tests now supply RoPE.
Configured architecture adds five integration tests. Optimizer settings add
32 validation cases; scheduler adds 26 tests for rates, boundaries, parameter
groups, and restoration of optimizer/scheduler state with matching updates.
Mixed precision and a full training/checkpoint loop are not yet tested or built.
