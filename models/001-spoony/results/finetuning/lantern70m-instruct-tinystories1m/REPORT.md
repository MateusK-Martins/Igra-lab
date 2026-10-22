# Normal instruction model -> TinyStories 1M

Initialized from the normal instruction SFT model, not the think-token model.
Original 8192-token vocabulary; 70,006,272 parameters. Plain story text + EOT,
no instruction wrapper or think markers. 1,000,000 stored training tokens,
999,424 supervised target tokens in full 2048-token windows; 61 updates.
TinyStories pinned revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64.
4,013 training stories; separate 50,000-token validation slice (209 stories).
Deduplicated by stripped text SHA256 across both splits; final stories truncated.

Validation loss: **2.675018 -> 2.088918**. Perplexity: **14.512616 -> 8.076171**.
AdamW3e-5, BF16, batch2/accumulation4, warmup5/cosine; no QAT.
All 492 tests pass. Checkpoint at step61 and external mirror have identical SHA256.
Backup directory: /home/mateus/Documents/Igra-lab-checkpoints/lantern70m-instruct-tinystories1m/.
Checkpoints include optimizer, scheduler, RNG, and data-loader state. Inference
export and source/config/data snapshot are also backed up.

Samples use greedy and seed42+index top-k40/temperature0.8, at most128 new tokens.
They are qualitative probes, not proof of general instruction or reasoning gains.

## greedy: Once upon a time, there was a little rabbit who

```text
 lived in a small house. He was very proud of himself and he was very proud of himself. He was very proud of himself and he was very proud of himself.
One day, he was very proud of himself. He was very proud of himself and he was proud of himself. He was proud of himself and he was proud of himself. He was proud of himself and he was proud of himself.
```

## greedy: Lily found a shiny key in the garden. She

```text
 was so excited to see the key. She was so excited to see the key.

The key was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so exc
```

## greedy: One day, a small robot lost its way. It

```text
 was a big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big,
```

## greedy: Tom wanted to help his friend, but

```text
 he was too late. He was so happy and he was so happy.
He was so happy and he was so happy. He was so happy and he was so happy. He was so happy and he was so happy.
```

## greedy: The little bird was afraid to fly. Then

```text
, the little bird was so excited!
The little bird was so happy! It was afraid and it was so happy! It was so excited! The little bird was so happy! The little bird was so happy! It was so happy! It was so happy! It was so happy! It was so happy! It was so happy! It was so happy! It was so happy! It was so happy! It was so happy! It was so happy! It was so happy! It was so happy
```

## greedy: User: Tell me a short story about a kind dragon.
Assistant: 

```text
Once upon a time, there was a little girl named Lily. She was very happy and loved to play with her. She was very excited and she was very happy. She was very happy and loved to play with her. She was very happy and loved to play with her.
Lily was very happy and loved to play with her. She was very happy and loved to play with her. She was very happy and loved to play with her. She was very happy and loved to play with her. She
```

## top_k: Once upon a time, there was a little rabbit who

```text
 lived in a small village. The rabbit was very kind and sad. She was happy to be the best friend.

One day, the rabbit had a big gag and she had a big ticket. She was so excited! She gave her bucket to the rabbit and gave her a big wedge. The rabbit asked, "Do you want to help you?!"
The rabbit came out and asked the rabbit. The rabbit said, "Let's go, to the bigger village!" The ra
```

## top_k: Lily found a shiny key in the garden. She

```text
 took her to the garden and looked around her.

One day, they went inside and said, "Get so big!" But then, they were very happy.
When they arrived at the garden, a smile came up. 
At the garden, the kitchen smiled and said, "Why are you here?" 
The smile was a pink. The kitchen smiled and said, "Would we have a shiny key?" 
They said, "Who are you now here?" 
Sudden
```

## top_k: One day, a small robot lost its way. It

```text
 was not so tired. The robot had a great time and it had not been so much fun at it.
```

## top_k: Tom wanted to help his friend, but

```text
 he was too scared, and he took the kitchen to the home. His friend, Candie, helped him make the kitcheny. He sang this to the family and the kitcheny was always happy. He decided to try to keep the kitcheny and give it to their friend.
With the kitcheny, Tom decided to play with his friend. He went to the kitchen to play with his friends. When he arrived, he picked up the kitcheny and the kitcheny. He made his friend's wonderful bag
```

## top_k: The little bird was afraid to fly. Then

```text
 the bird was ready to fly it and run away.
"Yes, The bird wants to fly. We will go the daylight, get some help. Did the bird fly?"
The bird was happy and didn't like the bird. It was not nice to fly. The bird wiped it and flew away.
"Yes, The bird wants to fly," said the bird. The bird wished to fly from the bird's home.
"Thank you," said the bird.
"No,
```

## top_k: User: Tell me a short story about a kind dragon.
Assistant: 

```text
"I'm sorry, but I can't do it to you!"
"But the dragon is still here!"
"I'm sorry, but I can't do it!"
Suddenly, a little girl came out. She was ready to be careful. She was very happy and frightened. She went to the bathroom to look for something. She saw it and wondered what it was.
Inside the bathroom, she woke up. She wanted to help it and found a new dragon. She tried to help
```

