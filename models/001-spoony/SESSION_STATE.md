Clean template/tide: standalone TideMixer/TideMixerImpl, same14M architecture/init/weights behavior. Old experimental architectures, tests, reports and historical docs removed. Shared training/data/SFT/cache utilities retained. No training launched. See README.md.

Follow-up cleanup: removed unused attention CacheEntry/ring-buffer/read_all paths
and fixed-decay scan helper; recurrent format-v1 snapshots remain compatible.
Replaced obsolete KV tests with recurrent persistence/validation/device tests.
419 tests and Ruff passed. Triton work remains on model/001-spoony-tide.
