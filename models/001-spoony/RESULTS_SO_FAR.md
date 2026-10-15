# FluxV experiments so far — No-K completion

The user stopped the remaining suite after No-K completed. No-K reached 9,000
updates and 18,429,440 target tokens from scratch. Tokenizer/data fingerprints,
settings, frozen source hashes and unchanged initialization were verified against
original FluxV; Lantern has identical data/settings/exposure. 638 tests and Ruff
passed before training. The original FFN 1024 was retained by explicit user choice.

## Completed results

All rows below completed the same 9,000 updates/18,429,440 target-token exposure.
A/E timing covers only their resumed phase; throughput uses that phase's actual
new tokens. Their phase durations must not be compared to complete-run durations.
Historical timings were measured in earlier sessions, not simultaneous benchmarks.

| Model | Parameters | Final loss | Perplexity | Measured min | Timed updates | Tokens/s | Peak CUDA GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original FluxV | 13,774,080 | 2.173892 | 8.7924 | 6.59 | 9,000 | 46,589 | 1.044 |
| A HeadNorm | 13,774,080 | 2.177169 | 8.8213 | 4.48 | 6,000 | 45,682 | 1.045 |
| B HeadNorm + matrix readout | 13,774,080 | 2.190446 | 8.9392 | 6.85 | 9,000 | 44,824 | 1.081 |
| C GlobalNorm + matrix readout | 13,774,080 | 2.187075 | 8.9091 | 6.77 | 9,000 | 45,364 | 1.080 |
| D head-axis MLP | 13,775,988 | 2.171528 | 8.7717 | 7.07 | 9,000 | 43,472 | 1.130 |
| E recurrent matrix memory | 14,211,456 | 2.039836 | 7.6893 | 35.04 | 8,000 | 7,792 | 1.044 |
| Lantern attention | 12,440,064 | 2.018320 | 7.5257 | 7.19 | 9,000 | 42,727 | 0.748 |
| 4-K tied Q | 16,446,720 | 2.169084 | 8.7503 | 15.84 | 9,000 | 19,396 | 1.777 |
| 4-K independent Q | 17,331,456 | 2.135759 | 8.4635 | 15.78 | 9,000 | 19,462 | 1.811 |
| No-K signed writes | 12,887,040 | 2.123176 | 8.3576 | 6.84 | 9,000 | 44,938 | 0.940 |

## What No-K reveals

No-K lowers loss by 0.050716 nats/token and perplexity by
4.95% against original FluxV, while removing
887,040 parameters (6.44%). Its cache is one 384-feature FP32 state
(1.5 KiB/sample/layer), half original FluxV's two 384-vectors (3 KiB).

The mechanism is simply S=g(x)*S_previous+tanh(W_V x), followed by
sigmoid(W_Q x)*RMSNorm(S). There is no K projection, K accumulator, K norm or
hidden dummy K state. All outer norms, FFNs, embeddings, residual behavior and the
existing learned token-dependent retention architecture remain unchanged.

This result shows the K cascade is not required to reach original FluxV's quality
on this recipe. It does not isolate K removal alone: direct signed tanh writes
replace sigmoid(V)*normalized(K), and K RMSNorm disappears. A plausible explanation
is that direct signed writes permit useful cancellation/update directions without
K preconditioning, but the experiment cannot identify that as the causal reason.
The smaller model's improvement makes 'more projections/state automatically help'
a poor explanation of the earlier vector results.

No-K is also 0.012583 nats/token better than four-K independent Q,
with 25.64% fewer parameters and roughly 2.31× its measured throughput.
The four-bank mechanism adds distinct projections, decays and key-query gating;
these results do not show that increasing bank count improves useful memory.

## What the earlier ablations reveal

- A vs original: head-wise normalization alone did not improve loss (+0.003276).
- B vs C: head-wise normalization under fixed matrix readout was slightly worse
  (+0.003371). Its effect resembles A vs original.
- C vs original: temporary matrix readout under global norms worsened loss
  (+0.013183). B vs A similarly worsened it (+0.013278).
- The 2×2 interaction contrast is only +0.000095 in this seed. There is no strong
  observed beneficial interaction between independent normalization and temporary
  matrix mixing. This does not prove such interactions cannot help at other budgets.
- D's small loss change (-0.002365) is inconclusive. Its MLP learned nonzero
  cross-head coupling, but identity dominated; coupling activity is not evidence
  of useful semantic specialization.
- Four-K independent Q improved on tied Q by 0.033324, consistent with query sharing
  being restrictive here. It also adds 884,736 parameters, so capacity is a confound.
  Both are now behind the simpler No-K model in final loss.
- E's persistent matrix memory remains the best completed attentionless model:
  loss 2.039836/PPL7.6893. It differs fundamentally from temporary matrix readouts,
  storing historical rank-one writes. Its state is 8192 elements per layer versus
  No-K's 384 (21.33×), and measured throughput is much lower. The result supports
  investigating richer memory representation, not claiming matrix readout alone
  helped or that the model has demonstrated retrieval/reasoning.

## Lantern remains the strongest reference here

Lantern's loss 2.018320/PPL7.5257 is still lower than No-K and E. It has 12,440,064
parameters, fewer than No-K, and uses causal GQA with window 256, RoPE, the same
width 384/six layers/FFN 1024/data/exposure. No-K's run took 6.84 min versus Lantern's
7.19 min; the historical session difference limits precise speed conclusions.
Lantern's inference KV storage grows with retained tokens, while No-K holds a
fixed 384-feature state per layer. Thus No-K offers simpler constant-state inference,
but the current tests do not establish a quality advantage over attention.

## Stability and learned forgetting

All initial/final No-K outputs and parameter gradients are finite. Training retained
the existing nonfinite-gradient check at every BF16 optimizer update. The following
statistics are from the fixed validation-batch probe, averaged across six layers:

- Mixer output RMS: 0.5169 → 0.4990.
- Un-clipped diagnostic gradient norm: 3.1325 → 1.7139.
- Learned token-retention mean: 0.90128; token variation std: 0.13973.

The final nonzero token variation confirms input-dependent forgetting on the sampled
batch. It does not establish which information is forgotten or whether gates carry
semantic roles. Complete per-layer/parameter statistics are in the initial/final
probes. Story samples and validation curves are retained, without treating them as
QA or reasoning evaluations.

## Preserved but incomplete experiments

- Two-K: implemented, 638 tests passed, memory/gradient preflight passed; stopped at
  last logged update 660 after validation 500. No 1000-update checkpoint existed.
  Partial event logs are retained; there are no resumable trained weights from this
  stopped run. It is excluded from final-budget comparisons.
- Simple diagonal state: implemented/tested/preflight; no complete training run.
- Signed-mass normalization: implemented/tested/preflight; no complete training run.

These cannot yet answer whether token-dependent retention beats fixed diagonal
retention, whether two K banks are preferable to four, or whether historical-mass
normalization can replace RMSNorm successfully. Their branches/configurations are
preserved. No trainer or queued runner remains, and temporary sleep inhibition
was released. Future training requires a new explicit request.

## No-K fixed-prompt generations

These use the inherited best-saved-checkpoint procedure, 64 new tokens max. Final
loss/PPL above uses the completed 9000-update state. Samples are illustrative only.

Prompt: Once upon a time, there was a little girl named Lily.

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, scary dog. The dog was barking and running around. Lily was scared of the dog, but she knew she had to be brave.
> As she walked, she saw a big dog. The dog was barking and running towards her.

Prompt: Tom found a small dog outside his house. The dog was

> Tom found a small dog outside his house. The dog was very happy and said, "Thank you, dog! You are a good dog."
> The dog wagged his tail and licked the dog's face. He wagged his tail and licked their faces. He was happy to see the dog and the dog.
> "Thank you, dog!" Lily said. "You are very

Prompt: One day, a boy lost his favorite toy. He felt

> One day, a boy lost his favorite toy. He felt sad and lonely. He wanted to play with the toy, but he could not find it. He looked under the bed, in the closet, and in the toy box, but he could not find it. He was sad and angry.
> Then, a kind girl named Lily saw the toy. She was very happy

All conclusions are single-seed observations at limited TinyStories exposure. No
statistical significance, general architecture-superiority or reasoning claim is made.
