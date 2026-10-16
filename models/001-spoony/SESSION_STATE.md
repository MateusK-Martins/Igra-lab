Tide/Lantern hybrid completed on model/001-spoony-tide-lantern-30m.
Pattern Tide,Tide,Lantern x3, width512, FFN1152,28,395,520params, six Tide/three GQA.
730tests/Ruff passed.14M branch model/001-spoony-tide-lantern-14m has completed
9000updates, loss1.983780/PPL7.270169, results commit497784f.
30M mixed+QA stages each6500updates/26,624,000padded positions. Mixed storyloss1.990089,
QA6.056534, EM0/F1.476465%; afterQA story2.769100, QA3.766497, EM.390625/F1 3.015468%.
See HYBRID_30M.md for controlled comparisons, cache limits, stability probes, samples,
data/settings/exposure/source hashes and checkpoint initialization verification.
Pretraining CUDA failed after suspend/resume; resumed6000 with optimizer/RNG/loader
state. Mixed timing approximate/first-attempt peak missing; restart values preserved.
QA-stage timing/peak measured directly. Sleep inhibition ended; no trainer remains.
No additional model tuning or training started. Reports/checkpoints preserved locally;
large checkpoints/data ignored. No push requested for these hybrids.
