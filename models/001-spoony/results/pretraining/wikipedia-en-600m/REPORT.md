# Lantern 70M — 600M-token Wikipedia results

Completed 36,621 updates and 599,998,464 supervised target tokens.
Final validation loss: **2.682159**; perplexity: **14.616614**.
Training took approximately 8 hours 27 minutes. Maximum recorded temperatures:
GPU 70 C; CPU 79 C. No health stop occurred.

The final resumable checkpoint is `step-00036621.pt`, saved locally and mirrored
outside the worktree. The GitHub release `lantern70m-wikipedia600m-v1` includes
the full checkpoint, inference export, tokenizer, training source snapshot and hashes.
Extending to a new corpus/budget requires explicit continuation configuration;
the existing strict resume checks intentionally reject changed dataset/schedule settings.

Samples below use the original six probes, greedy and seeded top-k 40,
temperature 0.8, up to 128 new tokens. They exhibit repetition and factual errors;
validation improvement does not establish reliable factual knowledge.

## greedy: The Solar System consists of

```text
 a single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-sided, single-
```

## greedy: Photosynthesis is the process by which

```text
 the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the photosynthesis of the phot
```

## greedy: The Roman Empire was

```text
 established in the Roman Empire. The Roman Empire was a part of the Roman Empire, and the Roman Empire was a part of the Roman Empire.

The Roman Empire was a part of the Roman Empire, and its territory was part of the Roman Empire. The Roman Empire was a part of the Roman Empire, and its territory was part of the Roman Empire. The Roman Empire was a part of the Roman Empire, and its territory was part of the Roman Empire. The Roman Empire was a part of the Roman Empire, and its territory was part of the Roman Empire.

The Roman Empire was a part of the Roman Empire,
```

## greedy: In mathematics, a prime number is

```text
 the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of points in the number of
```

## greedy: The Amazon River is

```text
 a tributary of the River Thames, which flows into the River Thames at the mouth of the river.

The river is a tributary of the River Thames, which flows into the River Thames at the mouth of the river.

The river is a tributary of the River Thames, which flows into the River Thames at the mouth of the river.

The river is a tributary of the River Thames, which flows into the River Thames at the mouth of the river.

The river is a tributary of the River Th
```

## greedy: A computer processor is

```text
 used to process the processor's output.

The processor is used to process the processor's output, and the processor's output is used to process the processor's output. The processor's output is used to process the processor's output, and the processor's output is used to process the processor's output.

The processor's output is used to process the processor's output, and the processor's output is used to process the processor's output. The processor's output is used to process the processor's output, and the processor's output is used to
```

## top_k: The Solar System consists of

```text
 a single pin, called a sitting sitting sitting-type, or a ribbon-pin, or a pin-sitting sitting-type. A sitting sitting sitting sitting sitting-type is a pin-sitting sitting sitting-type. It is not a sitting sitting sitting sitting sitting-type but a pin-sitting sitting sitting-type.

It is similar to the Solar System for Solar System, although its routing is often used to connect the Solar System
```

## top_k: Photosynthesis is the process by which

```text
 the acetyl group of the group can be identified. The acetyl group, known as the acetyl group, with a specialized organic group, is a group of the compounds.

Software
Software is used primarily for commercial applications like automotive stores and for commercial applications like caramel. Software is currently used in commercial applications such as the Adobe ReadTrace Linux and the Alto-Software Linux.

Software is not available in the same manner as the Acetyl-S
```

## top_k: The Roman Empire was

```text
 reunited between the Roman and Roman governors of Italy.

In 959, the Roman and Roman governors were reunited together with the Roman governor of the same name. During this period, the Roman-Born Roman Empire was divided between Rome and the Holy Roman Empire and Rome was divided between the Roman-Roman and the Roman empire until the end of the reign of Roman emperor Constantine the Great.

In 968, the Roman emperor Constantine the Great launched a raid of Rome that killed the emperor's sons, which would lead to his death. A Roman army was sent against Rome and the
```

## top_k: In mathematics, a prime number is

```text
 the number of the second-generation of the number of the third-generation of the number of non-inverted cipher types. The number of elemental cipher type types is a prime number; the number of elements in C, and the number of elemental cipher type types and the number of elemental ciphers.

In order to achieve this, the number of elemental ciphers on the C, the number of elemental ciphers on the C, and the number of elemental ciphers in the C, the number of elements in the C
```

## top_k: The Amazon River is

```text
 the largest in the Great Plains of Arizona and the largest in the Northeast.

The river was first described in 1866 as the Rio Alto River, and was established by Governor Charles H. Hurst in 1867.

See also

References

Further reading

Works cited

External links

Rivers of Tucson, Arizona
Rivers of Sacramento County, Arizona
Santa Clara River
Rivers of Arizona
```

## top_k: A computer processor is

```text
 the device of the "single tracker" that is designed to provide an electronic system, and is thus designed to interrupt the input of the device.

The computer system is designed to compensate for this effect, but can also be used to make a program that is too truly correct, and can be either a machine that is capable of interfering with the user's perceived "compensation". It can also be used to make a program that reduces the efficiency of a machine, and to make a program that gives it a better performance.

Processor

```

