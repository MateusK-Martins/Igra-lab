# Declarative model assembly guide

This document defines how to declare, validate, and assemble a language model.
It covers the implemented composition machinery. SwiGLU is implemented;
attention, normalization, and positional encoding remain upcoming work.

Current status: definition protocols, recursive validation, composition,
assembly, weight tying, forward/backward, and state-dictionary round trips are
tested. No editable architecture file or final Transformer exists yet.

Use the numbered steps below to resume pair programming. Complete one step,
review its behavior, and then proceed to the next.

## Agreed declaration

The editable model definition belongs in `configs/model/architecture.py`.
It is ordinary Python using dataclass definitions, with no external parser.

Working declaration syntax using implemented classes and an already trained
tokenizer (this is a composition example, not a Transformer architecture):

```python
definition = Model(
    tokenizer=tokenizer,
    input=Embedding(output_features=256),
    blocks=[
        Repeat(
            times=3,
            blocks=[
                Sequential(
                    blocks=[Linear(256, 512), Linear(512, 256)],
                    residual=True,
                ),
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

model = assemble(definition)
```

`Model` describes the architecture; `LanguageModel` is the constructed
`torch.nn.Module`. Importing a configuration must not download data or train
a tokenizer. A trained tokenizer is supplied or explicitly loaded when the
definition is created.

## Shape contracts

Every ordinary block has positive integer `input_features` and
`output_features`, plus a boolean `residual` parameter. Blocks transform the
last dimension while preserving batch and sequence dimensions:

```text
[B, T, input_features] -> [B, T, output_features]
```

A standalone block may change its feature width. The assembler checks that
its neighbors can connect; it must not silently insert projections.

The embedding is a special boundary:

```text
integer token IDs [B, T] -> vectors [B, T, embedding.output_features]
```

Its vocabulary dimension is obtained from `definition.tokenizer.vocab_size`.
It does not have an ordinary vector `input_features` field.

The LM head is the other special boundary:

```text
hidden states [B, T, lm_head.input_features] -> logits [B, T, vocab_size]
```

Its resolved output size must equal the tokenizer's actual vocabulary size.
Do not substitute the requested BPE vocabulary size for the actual learned
size. Forward returns logits; loss calculation belongs to training code.

## Composition rules

### Sequential

`Sequential(blocks=[...], residual=False)` executes children in order.
Its input width is the first child's input width and its output width is the
last child's output width. These are derived properties, not duplicated
editable fields. Empty sequential compositions are rejected initially.

For every adjacent pair:

```text
left.output_features == right.input_features
```

### Repeat

`Repeat(times=N, blocks=[...], residual=False)` repeats the complete child
sequence N times, in place. N must be a positive integer; children must be
non-empty.

For example:

```text
[A, Repeat(3, [B, C]), D] -> [A, B1, C1, B2, C2, B3, C3, D]
```

Each occurrence constructs fresh modules and parameters. Reusing a definition
does not imply sharing weights. Besides checking each child sequence, check
the connection from its last output to its first input when N > 1.

The repetition boundary uses its first child's input width and final child's
output width. A width-changing sequence can repeat once; repeating it more
than once requires compatible boundary widths.

### Residual

Residual is an exposed parameter on leaf blocks and composition definitions.
Its behavior is:

```python
output = input + branch(input)
```

Therefore its input and output widths must match. The branch must also
preserve batch and sequence dimensions. No automatic residual projection is
introduced.

A residual on Sequential wraps that complete sequence. A residual on Repeat
wraps all repetitions together. Residuals on children wrap individual children.
Repeat.build() constructs a runtime sequence directly; it preserves nested
wrappers rather than flattening away their scopes. Embedding and LM head are boundary components
without a residual option.

### LM head weight tying

If `tie_to_embedding=True`, the head's input width must equal the embedding's
output width. The head and embedding use the same actual Parameter object,
with shape `[vocab_size, embedding_width]`. Copying values is not weight tying.

The first implementation uses a bias-free head. Register the shared parameter
before constructing the optimizer, and ensure initialization does not reset
the same shared weight twice. Saving and reloading must restore the tying.

## Code organization

```text
configs/model/
    architecture.py       # Editable model declaration

src/model/
    __init__.py
    definitions.py        # Block/boundary protocols and concrete definitions
    assembler.py          # Model declaration, recursive validation, assemble()
    composition.py        # Residual module and build_block() helper
    feedforward.py        # Runtime SwiGLU feed-forward module
    language_model.py     # Public PyTorch model and forward boundary

tests/
    test_model_definitions.py
    test_model_validation.py
    test_model_composition.py
    test_model_assembly.py
    test_swiglu.py
```

The old `src/model.py` placeholder has been replaced by this package. The
`configs/model/architecture.py` file is planned. Component modules are added
as their internals are implemented.

Definitions satisfy structural protocols and own a build() method that returns
a fresh PyTorch branch. The assembler reads the protocol fields for validation,
recognizes Sequential and Repeat for recursive checks, and uses build_block()
to apply residual wrappers. No component registry is required.

## Numbered implementation steps

### 1. Define the declaration dataclasses

Status: completed, using separate boundary protocols and BlockDefinition.

Create the model package and `definitions.py`. Define Model, Embedding,
LMHead, Sequential, Repeat, and a minimal ordinary block contract. Use frozen
dataclasses for settings and accept the agreed list-based composition syntax.
Do not mutate those child lists during assembly.

Do not invent attention internals yet. Use a simple Linear definition when
we need a concrete block to exercise the machinery.

Check positive integer dimensions, boolean residual settings, positive repeat
counts, and non-empty compositions. Explicitly reject booleans as dimensions
or repeat counts, because Python treats bool as a subtype of int.

### 2. Resolve composition dimensions

Status: completed through Sequential/Repeat properties and recursive validation.

Implement recursive input/output feature resolution. Leaf dimensions are
declared; Sequential and Repeat dimensions are derived from their children.
The tokenizer supplies vocabulary dimensions at model boundaries.

Test nested compositions and avoid allocating PyTorch modules during this step.

### 3. Validate the whole model

Status: completed in Model.validate() and Model.validate_blocks(). Constructor
validation runs immediately and assemble() repeats it before allocating weights.

Walk the definition recursively. Validate every neighboring connection,
repeat boundary, and residual width. Connect the embedding to the first body
block, then the final body output to the LM head. An empty Model.blocks list
may connect embedding directly to the head, although child compositions must
not be empty.

Verify tying dimensions and final vocabulary width. Error messages should name
the definition path and expected/actual widths, for example:

```text
blocks[0].blocks[1]: expected input_features=256, received 512
```

Tests must include valid width changes, invalid neighbors, invalid residuals,
invalid repeated boundaries, and incompatible tied heads.

### 4. Expand Repeat without changing the declaration

Status: completed directly in Repeat.build(); a separate expansion-plan module
is not needed. It calls build_block() for each child on each repetition and
returns nn.Sequential. Nested groups remain registered modules, preserving
residual scopes. Each build() creates independent parameters. Settings and
child lists are not modified during construction.

### 5. Implement runtime composition

Status: sequential execution and residual wrapping completed and tested.
Attention execution context is still to be designed.

Create sequential execution with registered child modules (`ModuleList` or
equivalent), and a residual wrapper. Test them with small deterministic Linear
modules. Parameters must appear in `model.parameters()` and state dictionaries.

Define how execution context travels through compositions. Attention will
need positions and masks: choose one shared forward convention and pass that
context consistently. Do not store per-batch masks in architecture settings.

### 6. Build components and assemble the wrapper

Status: completed for current components in assemble() and LanguageModel.

Implement assembly by calling each definition's build() through build_block().
Construct a fresh module per occurrence, then wrap residual branches at the
declared scope. Build the
embedding and vocabulary head using the tokenizer's actual vocab size.

LanguageModel.forward maps token IDs `[B, T]` to logits `[B, T, V]` through
embedding, body, and head. Explicit input-rank and context-limit checks are not
implemented yet. Linear and SwiGLU exercise construction before attention.

### 7. Implement and verify weight tying

Status: completed; save/load tests rebuild from the definition before loading
the state dictionary, which restores the declared shared parameter relationship.

Assign the embedding Parameter to the head when requested. Test object identity,
gradient flow, independent weights when tying is disabled, and save/load
restoration. Check repeated body blocks have distinct Parameter objects.

### 8. Add the editable architecture file

Status: pending. Keep imports free of downloads or tokenizer training.

Create `configs/model/architecture.py` using the agreed declaration. Provide
the trained tokenizer explicitly when creating the definition, so importing
the file is possible before a tokenizer artifact exists. Keep numeric choices
in the editable definition and use named variables for shared dimensions.

The runnable entry point explicitly loads the tokenizer, creates the definition,
and calls assemble. The builder does not import experiment configurations.

### 9. Verify the composition machinery end to end

Status: completed for implemented components. Extend coverage as attention and
its execution context are added.

Use a tiny test tokenizer and small tensors. Verify logits shape, backward
gradients, module registration, repeat independence, residual behavior, tying,
and meaningful validation errors. Tests need no corpus download or GPU.

The current checks pass. Next implement the remaining Transformer components
and their block definitions, then declare a full architecture.

## Later decisions

Component-specific settings, normalization placement, positional encoding,
attention masking, cache support, and initialization policy must be designed
when implementing their corresponding modules. Arbitrary graphs, projected
residuals, and parameter sharing beyond embedding/head tying are extensions,
not requirements for this first assembler.
