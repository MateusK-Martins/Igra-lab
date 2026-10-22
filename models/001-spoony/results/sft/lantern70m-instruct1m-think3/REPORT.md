# Normal SFT followed by three-think-token continuation

Stage 1: 999,849 prompt+response+EOT tokens, 9,453 unique examples, 1,182 updates.
Final response validation loss 2.614870, perplexity 13.665443.

Stage 2: 999,995 tokens including markers, 2,364 unique examples selected as the
longest 25% of stage-1 instructions, cycled to 8,970 training records; 1,122 updates.
Instruction lengths are 30-550 tokens. All answers have three supervised think
markers; no reasoning text is supplied. Stage-1 neural weights are preserved
exactly, with one newly initialized tied vocabulary row. Stage-2 AdamW and
schedule start fresh. Final marker+answer validation loss 2.733928,
perplexity 15.393236. Validation has the same 839 held-out instructions.
This is not a controlled test of think markers alone: stage 2 also uses a more
restricted, repeated dataset and additional optimization, and its loss includes
marker targets. It does not establish a benefit from the markers.

Both final checkpoints include optimizer/scheduler/RNG state and are mirrored
under /home/mateus/Documents/Igra-lab-checkpoints/lantern70m-sft1m-{normal,think3}/.
Model exports, source/configs and selected data are also backed up.
Stage 2 final checkpoint and mirrored copy have identical SHA256 hashes.

## Matched samples

### greedy: What is the capital of France?

Normal:
```text
The capital of France is the capital of France.
```

Think continuation:
```text
<think><think><think>The capital of the country is Frères.
```

### greedy: What is 7 plus 5?

Normal:
```text
7 plus 5 is 7 plus 5.
```

Think continuation:
```text
<think><think><think>7 plus 5 is 7 plus 5.
```

### greedy: Explain photosynthesis in one sentence.

Normal:
```text
The photosynthesis of one sentence is photosynthesis.
```

Think continuation:
```text
<think><think><think>Photosynthesis is a process of photosynthesis in which the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photos
```

### greedy: Write a short story about a lost robot.

Normal:
```text
Once upon a time, there was a mysterious robot named Jake who was a loyal friend of his. He was a loyal friend of his, and he was a loyal friend of his. Jake was a loyal friend of his, and he was a loyal friend of his. Jake was a loyal friend of his, and he was a loyal friend of his. Jake was a loyal friend of his, and he was a loyal friend of his. Jake was a loyal friend of his, and he was a loyal friend of his. Jake was
```

Think continuation:
```text
<think><think><think>The robot was awoken by the same name and had a heart-warming heart. He was determined to make it through and he was determined to make it through. He was determined to make it through and he was determined to make it through. He was determined to make it through and he was determined to make it through. He was determined to make it through and he was determined to make it through. He was determined to make it through and he was determined to make it through.
```

### greedy: Pretend you are a pirate. Introduce yourself.

Normal:
```text
I am a pirate.
```

Think continuation:
```text
<think><think><think>I am a pirate.
```

### greedy: List three fruits.

Normal:
```text
1. Cocoa, cocoa, and sugar.
2. Cocoa, cocoa, and sugar.
3. Cocoa, cocoa, and sugar.
4. Cocoa, cocoa, and sugar.
5. Cocoa, cocoa, and sugar.
6. Cocoa, cocoa, and sugar.
7. Cocoa, cocoa, and sugar.
8. Cocoa, cocoa, and sugar.
9. Cocoa, coc
```

Think continuation:
```text
<think><think><think>1. Fruit 
2. Fruit 
3. Fruit 
4. Fruit
```

### top_k: What is the capital of France?

Normal:
```text
The capital of France was Vendôme.
```

Think continuation:
```text
<think><think><think>The city is the administrative capital of the Pyrenees.
```

### top_k: What is 7 plus 5?

Normal:
```text
7 plus 5 is 7 plus 4.
```

Think continuation:
```text
<think><think><think>7 plus 5: 1-2, 3-4, 4-5, 5-6, 6-8, 6-9, 9-12
```

### top_k: Explain photosynthesis in one sentence.

Normal:
```text
Our photosynthetic and its potential use for photosynthesis is to produce a photosynthesis of two sentences in a sentence.
```

Think continuation:
```text
<think><think><think>Flipping sentence.
```

### top_k: Write a short story about a lost robot.

Normal:
```text
Once upon a time, the robot had become a devoutly belligerent and became an ideal choice for him. Despite being able to do so, his task would be to make a sense of what he'd do to make his own unique. Once upon a time, he was finally able to complete his education and he was always able to make the best of his students.
```

Think continuation:
```text
<think><think><think>On their way, the robots hurled their robots into a crisp, cordial process to pull it out of their bounds. The robots ridden inside the robotic robot were able to fake the robotic robot's death. They watched as the robots came across a robot who was able to make the robotic robot into a deep-seated robot. On one hand, they were determined to make the robotic robot a better world, filled with the world's most important. The robots were able to make the rob
```

### top_k: Pretend you are a pirate. Introduce yourself.

Normal:
```text
They all have the ability to fly away, without any havoc.
```

Think continuation:
```text
<think><think><think>I am a pirate, so I am unable to access the dough.
```

### top_k: List three fruits.

Normal:
```text
The five fruits are cherries and spices.
```

Think continuation:
```text
<think><think><think>1. Fruit 
2. Grain 
3. Grain 
4. Grain 
5. Grain 
6. Grain 
7. Grain 
9. Grain
```

