# Lantern 70M vs TokenMLP 70M

Both models used the same tokenizer, pinned Wikipedia training and validation data, seed 42, context 2048 and full one-pass exposure. Both completed 6,104 updates and processed 99,999,744 target tokens. The TokenMLP model replaces all nine FFNs, sharing two width-64 token tables and the 768→64→768 projections across layers.

| Model | Parameters | Validation loss | Perplexity | Updates | Target tokens | Training time |
|---|---:|---:|---:|---:|---:|---:|
| Lantern 70M | 70,006,272 | 3.1808 | 24.07 | 6,104 | 99,999,744 | ~85 min (approx.) |
| TokenMLP 70M | 20,723,200 | 3.4964 | 33.00 | 6,104 | 99,999,744 | 56.1 min (measured) |

TokenMLP has 70.4% fewer parameters and 0.3156 higher validation loss (nats) in this matched single-seed run. Greedy samples below use identical prompts and a 64-token continuation cap. These are illustrative probes; both models repeat, and the samples do not establish factual accuracy or instruction-following ability.

| Prompt | Lantern 70M | TokenMLP 70M |
|---|---|---|
| Once upon a time, there was a little girl named Lily. | Once upon a time, there was a little girl named Lily.<br><br>In the first season, Lily was a regular in the first season of the season, and was a regular in the first season. Lily was a regular in the first season, but was not a regular in the second season.<br><br>In the second season, Lily was a regular in the first | Once upon a time, there was a little girl named Lily.<br><br>The name of the name is derived from the name of the name of the name of the name of the name of the name of the name of the name of the name of the name of the name of the name of the name of the name of the name of the name of the name of the name of |
| Tom found a small dog outside his house. The dog was | Tom found a small dog outside his house. The dog was a large dog, and the dog was a dog, and the dog was a dog.<br><br>The dog was a dog, and the dog was a dog, and the dog was a dog. The dog was a dog, and the dog was a dog, and the dog was a dog. The dog was a | Tom found a small dog outside his house. The dog was found in the family home of the family home of the family home of the family home of the family home of the family home of the family home home home home home home home home home home home home home home home home home home home home home home home home home home home home home home home home home home home home |
| One day, a boy lost his favorite toy. He felt | One day, a boy lost his favorite toy. He felt that he was not afraid of the soul, but he was afraid of the soul.<br><br>In the summer of 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17 | One day, a boy lost his favorite toy. He felt that the boy was a good friend of the family, and that the boy was a friend of the family, who had a good friend, who had a good friend, who had a good friend, who had a good friend, who had a good friend, who had a good friend, who had a good friend, |

Run artifacts: [`training-report.json`](training-report.json), [`samples.json`](samples.json), and [`lantern-baseline-samples.json`](lantern-baseline-samples.json).
