# T16 template continuation

Local template/t16 is the new starting template; template/tide is preserved.
Promoted the existing 30M hybrid architecture from model/001-spoony-t16-50m
at 5200a64: width 512, nine layers, SwiGLU 1152, two Tide layers per one
causal A16 RoPE attention layer. Standalone TideMixerImpl and restored
attention/KV storage exclude experiment-only mechanisms and reports.
Template data/training defaults remain starting settings, not a reproduction
of the source mixed-corpus/QA pipeline. No training was launched.

Exact seeded state dictionaries, RNG and FP32 logits matched the source with
vocabulary 8192, seed 42 and matched CPU threads. Parameter count at that
vocabulary is 26,527,744. Recurrent cache IDs are renamed; recreate old caches.
Historical full training manifests cannot resume against changed source/configs.

505 tests passed, including attention/RoPE references, mixed-cache snapshot
continuation, recurrent tests and shared training/data/SFT checks. Ruff lint
and formatting passed. The original working checkout was untouched.
