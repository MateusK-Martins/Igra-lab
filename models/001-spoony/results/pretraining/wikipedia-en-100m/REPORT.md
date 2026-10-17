# Lantern 70M: 100M-token Wikipedia run

Completed 6,104 updates over 99,999,744 supervised targets (one epoch).

Final batch training loss: 2.963090. Held-out validation loss: 3.180789; perplexity: 24.065746.

## Qualitative samples

All six fixed prompts were decoded greedily and with temperature 0.8 / top-k 40, up to 128 new tokens. No repetition penalty was applied. Seeds, model hash and full continuations are in `samples.json`. Reproduce from this model directory with `.venv/bin/python -m scripts.sample` while the local model bundle is available.

Greedy outputs repeatedly loop and sampled outputs contain factual errors and topic drift. These probes do not demonstrate reliable factual knowledge. The model learned aspects of Wikipedia prose, but further work is needed before treating it as a knowledge model.

### greedy: The Solar System consists of

```text
The Solar System consists of the Solar System, which is a part of the Solar System.

The Solar System is a part of the Solar System, which is a part of the Solar System. The Solar System is a part of the Solar System.

The Solar System is a part of the Solar System, which is a part of the Solar System.

The Solar System is a part of the Solar System.

The Solar System is a part of the Solar System.

The Solar System is a part of the Solar System.

The Solar
```

### greedy: Photosynthesis is the process by which

```text
Photosynthesis is the process by which the tetrahedral is the process of the tetrahedral.

The tetrahedral is the most important of the tetrahedral, which is the most important of the tetrahedral. The tetrahedral is the most important tetrahedral in the Tetrahedral.

The tetrahedral is the most important tetrahedral in the Tetrahedral.

The tetrahedral is the most important tetrahedral in the Tetrahedral.

The tetrahedral is the most important tetrahedral in the Tet
```

### greedy: The Roman Empire was

```text
The Roman Empire was the first Roman Empire to be the first Roman Empire to be the first Roman Empire to be the first Roman Empire to be the first Roman Empire to be the first Roman Empire to be the first Roman Empire to be the first Roman Empire to be the second Roman Empire.

The Roman Empire was the first Roman Empire to be the first Roman Empire to be the first Roman Empire to be the first Roman Empire to be the first Roman Empire to be the first Roman Empire to be the first Roman Empire to be the first Roman Empire to be the second Roman Empire. The Roman Empire was the first Roman Empire to be the first Roman Empire to be
```

### greedy: In mathematics, a prime number is

```text
In mathematics, a prime number is the same as the number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of number of
```

### greedy: The Amazon River is

```text
The Amazon River is a tributary of the Amazon River, which is the source of the Amazon River.

The Amazon River is a tributary of the Amazon River, which is the source of the Amazon River.

The Amazon River is a tributary of the Amazon River, which is the source of the Amazon River.

The Amazon River is a tributary of the Amazon River, which is the source of the Amazon River.

The Amazon River is a tributary of the Amazon River, which is the source of the Amazon River.
```

### greedy: A computer processor is

```text
A computer processor is a computer processor.

The computer processor is a computer processor, which is a computer processor. The computer processor is a computer processor, which is a computer processor. The computer processor is a computer processor, which is a computer processor. The computer processor is a computer processor, which is a computer processor. The computer processor is a computer processor, which is a computer processor. The computer processor is a computer processor, which is a computer processor. The computer processor is a computer processor, which is a computer processor.


```

### top_k: The Solar System consists of

```text
The Solar System consists of the Solar System and the Solar System, which was the first major power plant to be installed on the Solar System. The Solar System provides 3.1 and 2.6 and is available for powered by the Solar System and Solar System.

The Solar System and Solar System (SDDR) also uses the Solar System and Solar System (MEDR), which is a monitor from the Solar System for the Solar System and Solar System.

Subsidiomedes 
Together with the Solar System and Sol
```

### top_k: Photosynthesis is the process by which

```text
Photosynthesis is the process by which the sperm cells are tolerable and are methane.

See also 
Genetic and Set-molecular phosphorylation

References

 
General
Calcium
```

### top_k: The Roman Empire was

```text
The Roman Empire was the second-tier title in the Roman Empire. In the Roman Empire the emperor was the emperor of Emperor Xuanzuan, with the second-tier title and the fourth-tier title of king. He became king of the Byzantine Empire in the 12th century, and had to have been the first emperor of Byzantine control. In 1430–14, when the Roman Emperor of Rome was created, the emperor and Emperor of Rome the Emperor of Rome was created, and the Emperor of Rome was crowned on the capital, which would be the most important emperor of Rome. In the 14th century, the Emperor of
```

### top_k: In mathematics, a prime number is

```text
In mathematics, a prime number is the same as a one or more different one, but that is the case of a person in the opposite case, for each group which is known by a noun, the same as the one, and for each group.

See also

Notes

References 

Human mitigation
Kurturing (math)
```

### top_k: The Amazon River is

```text
The Amazon River is the only waterway that can be accessed via the Amazon River.

Lands

In the United States, the Amazon Lake is most of the waterway that comes from the Amazon River, which is the first  of waterway to be in the Amazon.  The mouth is an all-new waterway that provides water water to a large waterway, so the waterway would be picnicked.  Lands and other waterways can be reached by water-finding in the park.  The waterway will generally be the largest waterway, so the waterway will
```

### top_k: A computer processor is

```text
A computer processor is the primary source of a computer.

In many languages, in a way, the computer processor has a function (gas, but not a function) and can also be used to get the tool.  In a way that can be used to transmit a device's function (also called an oral function).  The software may also be used to transmit a user's user's path.

Radio and video

The "Radio" can feature a computer system that can also transmit a program from a computer to a computer or from a computer. The system usually forms a set
```
