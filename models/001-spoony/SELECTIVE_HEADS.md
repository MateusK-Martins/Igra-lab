# Selective-write No-K heads: specialized

Standard reference is No-K + independent write gate + global RMSNorm (dcebd87,
recipe97f48aa), loss2.090822/PPL8.091566. Squared-retention model is not the standard.

Four96-feature heads, fixed384-element FP32 recurrent state per layer.
S_h = sigmoid(W_g_h*x+b_g_h)*S_h_prev + sigmoid(W_I_h*x+b_I_h)*tanh(W_V_h*x)
r_h = sigmoid(W_Q_h*x)*HeadRMSNorm(S_h)
Output is concatenated r_h; existing residual wrapper adds x exactly once.
Original full-width projections are row partitions (each head sees all384 inputs).
Retention/write decisions already independent featurewise in standard No-K.
New mechanism changes normalization domains; no new projection, attention,
routing, extra mixing, recurrent matrix, cross-head competition or token loop.
Original affine scan processes all384 features together unchanged.
Per-feature learned RMSNorm gains and epsilon1e-6 preserved, no gain sharing.
Parameter count13,774,080, identical to standard. Cache1.5KiB/sample/layer,
independent of sequence length. Four-head cache IDs distinguish global baseline.

Uniform A: all g_init0.99, i_init0.5, matching standard reference.
Specialized B: g_init(0.90,0.95,0.98,0.995), i_init(0.20,0.10,0.05,0.02).
Gate weights start zero; biases are logits of declared probabilities repeated96.
Bias addition/sigmoid and recurrent accumulation remain FP32 under BF16.
All gates learned from full current input after training. Only these declared
biases differ between A/B; common initial parameters and RNG streams match.
No claim that initial biases guarantee distinct semantic memories. RMSNorm
largely cancels uniform write scale within a head except for epsilon, so lower
initial write gate is not by itself evidence of lasting selectivity.

Same frozen TinyStories tokenizer/data30k train/256validation docs, seed42,
context256,batch8,BF16,FFN1024,width384,sixlayers,AdamW3e-4,warmup300,
9000updates/target18429440tokens,validation500,checkpoint1000.
Fresh independent runs; same loader seed/order. No budget or scan changes.

675 tests per branch passed, including independent FP64 sequential output and
parameter/input-gradient references with multi-feature heads, headnorm group
isolation, initialization/RNG, causality/gradcheck, residual once, full/chunk/
single-token caching/save/load/reset/constant memory, BF16 precision and finite
gradients, long/zero sequences, incompatible configurations and parameter count.
Ruff lint/format passed. Benchmarks/probes discarded before fresh training.
