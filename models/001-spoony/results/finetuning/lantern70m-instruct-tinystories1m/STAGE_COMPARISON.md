# Matched samples: pretrained, instruction-tuned, TinyStories-tuned

Three checkpoints: original Lantern70M after 600M Wikipedia tokens; then normal
1M-token instruction SFT; then 1M-token TinyStories fine-tuning from that normal
instruction model. No think-token experiment is used here. Same 8192-token
vocabulary, six prompts, greedy and top-k40/temperature0.8 decoding, 128 new-token
cap. Sampling uses seed42+prompt index for every model. Continuations are unedited.
These are qualitative probes; they are not a factual or story-quality benchmark.

## Prompt

```text
Once upon a time, there was a little rabbit who
```

### pretrained600m, greedy

```text
 was afraid of the boys. The boys were not allowed to wear the rabbit's uniform, but they were allowed to wear the rabbit's uniform.

The boys were not allowed to wear the rabbit's uniform, but they were allowed to wear the rabbit's uniform. The boys were allowed to wear the rabbit's uniform, and the boys were allowed to wear the rabbit's uniform. The boys were allowed to wear the rabbit's uniform, and the boys were allowed to wear the rabbit's uniform.


```

### instruction1m, greedy

```text
 lived in the area. He was a great friend of the local gentleman, who was a great friend of the local gentleman. He was a great friend of the local gentleman, who was a great friend of the local gentleman. He was a great friend of the local gentleman, who was a great friend of the local gentleman. He was a great friend of the local gentleman, who was a great friend of the local gentleman. He was a great friend of the local gentleman, who was a great friend of
```

### tinystories1m, greedy

```text
 lived in a small house. He was very proud of himself and he was very proud of himself. He was very proud of himself and he was very proud of himself.
One day, he was very proud of himself. He was very proud of himself and he was proud of himself. He was proud of himself and he was proud of himself. He was proud of himself and he was proud of himself.
```

### pretrained600m, top_k

```text
 was a bit of an ink for fit in with the water. It was the mildly popular and unforgiving of the water. She was often mistaken for a baker but he died soon after the day.

The story of the water is narrated by Sylvia Cassidy, who is the only man left in her life. The book contains an epic of "How to Never Stop the Water Different Course", and a brief version of the book is part of the Ace Canyon Book.

References

1941 British
```

### instruction1m, top_k

```text
 lived in the area of the present-day Walsingham County, New York. He was an uncle of the famous English rabbies who lived in the area.
```

### tinystories1m, top_k

```text
 lived in a small village. The rabbit was very kind and sad. She was happy to be the best friend.

One day, the rabbit had a big gag and she had a big ticket. She was so excited! She gave her bucket to the rabbit and gave her a big wedge. The rabbit asked, "Do you want to help you?!"
The rabbit came out and asked the rabbit. The rabbit said, "Let's go, to the bigger village!" The ra
```

## Prompt

```text
Lily found a shiny key in the garden. She
```

### pretrained600m, greedy

```text
 was also the first to use the garden's garden to create a new garden.

The garden was named after the garden's founder, the poet and playwright, and was named after the garden.

The garden was named after the garden's founder, the poet and playwright, and was named after the garden's garden.

The garden was named after the garden's founder, the poet and playwright, and was named after the garden's garden.

The garden was named after the garden's founder, the poet
```

### instruction1m, greedy

```text
 was able to use her newfound knowledge and knowledge to help her find her true identity. She was able to make her way through the garden and eventually became a successful artist. She was able to make her way through the garden and eventually became a successful artist. She was able to make her way through the garden and eventually became a successful artist.
```

### tinystories1m, greedy

```text
 was so excited to see the key. She was so excited to see the key.

The key was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so excited! She was so exc
```

### pretrained600m, top_k

```text
 would later say that she had a great love for him, but she was an unwitting woman.

In 1789 there were 56 boys and 17 girls in the garden with a lot of pigs. She was buried with her mother-in-law as her maid of honour, a woman who was already married to a widower.

The garden was then built with a carved pool that gave her its name.

References

External links

Biographical notes

17th-century English women
15th-century English people
15th
```

### instruction1m, top_k

```text
 discovered that it was a great place to stay afraid of the food they need, such as baking, storing all of the food from the fountain. She also found a unique way to stay inside of the garden, but this was a way to avoid the baking process. They found the shiny key in the garden and found a beautiful atmosphere.
```

### tinystories1m, top_k

```text
 took her to the garden and looked around her.

One day, they went inside and said, "Get so big!" But then, they were very happy.
When they arrived at the garden, a smile came up. 
At the garden, the kitchen smiled and said, "Why are you here?" 
The smile was a pink. The kitchen smiled and said, "Would we have a shiny key?" 
They said, "Who are you now here?" 
Sudden
```

## Prompt

```text
One day, a small robot lost its way. It
```

### pretrained600m, greedy

```text
 was the first robot to be destroyed by a robot.

The robot was destroyed by a robot, but it was destroyed by a robot. The robot was destroyed by a robot, but it was destroyed by a robot.

The robot was destroyed by a robot, but it was destroyed by a robot.

The robot was destroyed by a robot, but it was destroyed by a robot.

The robot was destroyed by a robot, but it was destroyed by a robot.

The robot was destroyed by a robot,
```

### instruction1m, greedy

```text
 was a small robot who had been living in a small town, and it was a small robot who had been living in a small town. The robot was a small robot who had been living in a small town, and it was a small robot who had been living in a small town. The robot was a small robot who had been living in a small town, and it was a small robot who had been living in a small town. The robot was a small robot who had been living in a small town, and it was a small robot who had been living in a small town
```

### tinystories1m, greedy

```text
 was a big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big, big,
```

### pretrained600m, top_k

```text
 was destroyed in a fire, but the robots were later repaired and removed.

See also
List of robots

References

External links

Rock the Possible The Wizard: The Real World of Robots, by Paul O'Grady

Rock operas
Rock operas
Rock operas
1978 in the United States
1978 establishments in the United States
1978 establishments in the United States
1978 disestablishments in the United States
1978 sports television series debuts
1978 sports television series endings
```

### instruction1m, top_k

```text
 was determined to give it a better plan. It was able to take on the form of a robot of the same type and its purpose. Finally, it was able to take on the form of a lion. This robot was a great success and the robot's loyalty was no longer a good foe of its kind.
```

### tinystories1m, top_k

```text
 was not so tired. The robot had a great time and it had not been so much fun at it.
```

## Prompt

```text
Tom wanted to help his friend, but
```

### pretrained600m, greedy

```text
 he was too late to do so.

In the summer of 2006, he was hired by the New York-based company, the New York-based company, to work with the New York-based company, the New York-based company, to help with the development of the New York-based company.

In 2007, he was hired by the New York-based company, the New York-based company, to work with the New York-based company, the New York-based company, to develop and develop new designs for the New York-based company.

In 2008, he was hired by the
```

### instruction1m, greedy

```text
 he was too late to take up the job. He was determined to make the decision and he was given the job. He was determined to make the decision and he was given the job. He was determined to make the decision and he was given the job. He was determined to make the decision and he was given the job. He was determined to make the decision and he was given the job. He was determined to make the decision and he was given the job. He was determined to make the decision and he was given the job. He was determined to make the decision and he was given the job. He was given the job and
```

### tinystories1m, greedy

```text
 he was too late. He was so happy and he was so happy.
He was so happy and he was so happy. He was so happy and he was so happy. He was so happy and he was so happy.
```

### pretrained600m, top_k

```text
 he was too busy at his job.

Dennis and Leroy made an appearance in the series following the episode that was being played on, and they were not actually seen in the episode, but the episode was later cancelled.

Final season

Hosts

Native mythology
Leroy was first introduced as a fox hunting cat, and grew up in the middle of a hunting family. He grew up in one of the city's tall forests, and was named after the cat of the hunt.

Hosts
 Leroy has a
```

### instruction1m, top_k

```text
 he was too late in a game and he decided to take the lead and go on to a three-game brawl where he met a friend before he could hesitate to stop him from getting to know he is in the same game. Once again, he was still in the same game and began working hard at the same time in the same game. As the situation worsened, he got in trouble and he was so sick, as he wouldn't make it and was too late to get back to his friends. After a few moments, he was finally saved by a friend
```

### tinystories1m, top_k

```text
 he was too scared, and he took the kitchen to the home. His friend, Candie, helped him make the kitcheny. He sang this to the family and the kitcheny was always happy. He decided to try to keep the kitcheny and give it to their friend.
With the kitcheny, Tom decided to play with his friend. He went to the kitchen to play with his friends. When he arrived, he picked up the kitcheny and the kitcheny. He made his friend's wonderful bag
```

## Prompt

```text
The little bird was afraid to fly. Then
```

### pretrained600m, greedy

```text
, the bird was afraid to fly, and the bird was afraid to fly. The bird was afraid to fly, and the bird was afraid to fly. The bird was afraid to fly, and the bird was afraid to fly. The bird was afraid to fly, and the bird was afraid to fly. The bird was afraid to fly, and the bird was afraid to fly. The bird was afraid to fly, and
```

### instruction1m, greedy

```text
, the bird was caught by a large bird, who was able to fly. The bird was able to fly, but the bird was able to fly. The bird was able to fly, but the bird was able to fly, and the bird was able to fly. The bird was able to fly, but the bird was able to fly, and the bird was able to fly. The bird was able to fly, but the bird was able to fly, and the bird was able to fly. The bird was able to fly, but the bird was able
```

### tinystories1m, greedy

```text
, the little bird was so excited!
The little bird was so happy! It was afraid and it was so happy! It was so excited! The little bird was so happy! The little bird was so happy! It was so happy! It was so happy! It was so happy! It was so happy! It was so happy! It was so happy! It was so happy! It was so happy! It was so happy! It was so happy! It was so happy! It was so happy
```

### pretrained600m, top_k

```text
 the bird was chased by a local pigeon, but the bird came out of the bark and tried to fly, despite the little bird's fate.

Meanwhile, the bird was to fly and was to fly, but he was then able to fly, as the bird was to fly. After furling his tail, he was not able to fly.

How he drove, he steadfastly went for a bird, when he was in the water. He steadfastly took the bird in the water, but
```

### instruction1m, top_k

```text
 the bird was pinned by frogs and the bird was struck by a powerful boost, which had a powerful and cunning sound. The bird came back, and the bird was able to fly. The birds were so out of reach that they had been able to fly by slowly, and the bird was able to fly quickly, but the bird was able to fly by fast with his flammable canine. The bird was able to fly by the stick for long, so the birds were able to fly from the ground, and the bird was able to fly
```

### tinystories1m, top_k

```text
 the bird was ready to fly it and run away.
"Yes, The bird wants to fly. We will go the daylight, get some help. Did the bird fly?"
The bird was happy and didn't like the bird. It was not nice to fly. The bird wiped it and flew away.
"Yes, The bird wants to fly," said the bird. The bird wished to fly from the bird's home.
"Thank you," said the bird.
"No,
```

## Prompt

```text
User: Tell me a short story about a kind dragon.
Assistant: 
```

### pretrained600m, greedy

```text

The story of a young man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who is a man who
```

### instruction1m, greedy

```text
The dragon was a beautiful dragon who was a great dragon. He was a great dragon, but he was a great dragon. He was a great dragon, but he was a great dragon. He was a great dragon, but he was a great dragon. He was a great dragon, but he was a great dragon. He was a great dragon, but he was a great dragon. He was a great dragon, but he was a great dragon.
```

### tinystories1m, greedy

```text
Once upon a time, there was a little girl named Lily. She was very happy and loved to play with her. She was very excited and she was very happy. She was very happy and loved to play with her. She was very happy and loved to play with her.
Lily was very happy and loved to play with her. She was very happy and loved to play with her. She was very happy and loved to play with her. She was very happy and loved to play with her. She
```

### pretrained600m, top_k

```text
 A frog named Hunt.
Protestant:  A frog named Hunt.
Elder:  A frog named Hunt.

The following is a list of different versions of Hunt's story, from the original original in The Adventures of Hunt (1930), which was published by Marvel in 1937.

References

External links
 Bird of Tears at MarvelWeb

Characters created by David Shirer
Comics characters introduced in 1938
Marvel Comics characters who use magic
Marvel Comics characters who use
```

### instruction1m, top_k

```text
A small butterfly dragon is an exciting and adventurous species that is so powerful that it is the only people with the ability to fly. They must fly in the night sky to the ground and find themselves in a new place. They are always out on a journey to reach a better place, and they must take in a place to get to the top of the sea. For a short time, they have to stay in the world to find the best and wedlock. They will never take out anything but the best and welcome them to the world, as well as the best
```

### tinystories1m, top_k

```text
"I'm sorry, but I can't do it to you!"
"But the dragon is still here!"
"I'm sorry, but I can't do it!"
Suddenly, a little girl came out. She was ready to be careful. She was very happy and frightened. She went to the bathroom to look for something. She saw it and wondered what it was.
Inside the bathroom, she woke up. She wanted to help it and found a new dragon. She tried to help
```

