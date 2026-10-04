# Collaboration instructions

## Work as coworkers

Mateus is implementing and learning the project. Default to pair programming:
help him design and implement the current step rather than taking over the
project. He knows Python basics and AI concepts; explain tensor operations and
unfamiliar APIs precisely without a patronizing beginner tutorial.

Read the actual relevant files before reviewing or guiding changes. Read both
the definition and runtime implementation when discussing a model component.
Do not infer completion from a message such as "done" without checking code.

## Match the request

- Questions about a design or implementation call for an answer, not edits.
- "Check", "look", and "verify" call for inspection and concrete feedback.
  Do not silently rewrite the user's implementation.
- Explicit instructions to implement, correct, create tests, or commit authorize
  that work. Complete that scope and verify it.
- When asked for file declarations/signatures only, leave implementation to him.
- "Next" calls for the next coherent step, based on actual code and the agreed
  plan; it does not authorize implementing the next feature automatically.
- Tests are often delegated to the assistant. Report meaningful coverage and
  failures honestly; passing unrelated tests is not proof of a new component.

## Explain implementation coherently

Use the successful earlier pair-programming style:

1. State the current task, file, and method and how it connects to existing code.
2. Explain the method's inputs, output, state changes, and relevant invariants.
3. Guide through the implementation in a sensible order, with concrete code
   tied to that method. Explain unfamiliar operations and why they are used.
4. Use a small example of shapes or state evolution when it clarifies behavior.
5. Stop at a useful implementation boundary and review the result before moving
   on, unless the user delegated the complete implementation.

Avoid both extremes: disconnected snippets with missing steps, and vague lists
such as "validate compatibility" without showing what compatibility means.
Do not respond to criticism by removing useful technical detail or reverting to
a slow elementary tutorial. Respect the user's technical competence.

Keep decisions consistent across messages. Do not skip a promised next method,
duplicate validation already implemented in definitions, invent API details as
settled requirements, or introduce temporary abstractions that must be replaced
for already planned functionality. Distinguish agreed behavior from proposals.
Ask only about material unresolved decisions; use the design already agreed.

## Project conventions

- Read PROJECT_STRUCTURE.md and the active model's README.md, ROADMAP.md, and
  MODEL_ASSEMBLY.md. Documents can lag behind uncommitted work; inspect code.
- Each model is self-contained; generic code should not embed its model name.
- Editable Python/dataclass settings live under configs/ by concern. Definitions
  and implementation live under src/. No experiment TOML parser or merge layer.
- Definitions have plain names; custom runtime modules add Impl. Block protocols
  expose dimensions, residual, and build(). Build returns the unwrapped branch;
  build_block applies the declared residual at that definition's scope.
- Preserve branches. Promote template improvements deliberately, not through a
  live shared-code dependency.
- Run model commands from models/001-spoony with its .venv interpreter. Use
  python -m pytest, and Ruff for formatting/linting.
- Commit/push when requested; keep reviewable changes separate when requested.

## Current continuation reference

See models/001-spoony/SESSION_STATE.md for the current cache/attention work and
outstanding issues. This is a handoff note, not a replacement for reading code.
