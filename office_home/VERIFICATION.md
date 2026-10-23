# Office-Home verification snapshot

Office-Home source fingerprint: `273d238783d9b723e0e74604bad856b3209e485d4af8b4115aab5c82698bb95b`.

Unchanged PACS source fingerprint: `583fc84119e4d5ff16b323e91d446f394bd824eb1f400a94ee8bfdadff9ad4e2`. All existing tracked files remain unchanged.

| Check | Actual result | Evidence |
|---|---|---|
| Existing plus Office-Home tests | 81 passed; zero errors, failures, or skips | `verification/unit_tests.json` |
| Original-resolution images | All 15,588 verified, fully decoded, and hashed | `verification/dataset_preparation.json` |
| Original imported-byte identity | Pinned canonical collection digest checked during preparation and loading | `references/dataset_sources.json` |
| Class mapping | All 65 classes and final label 64 exercised | Tests and pretrained smoke |
| Full artificial-image pipeline | All three ResNet-18 arms trained, selected, saved, and evaluated | `verification/cpu_smoke.json` |
| Pretrained 224px steps | Finite CPU update for all methods; 11,209,857 parameters per arm | `verification/pretrained_cpu_smoke.json` |
| Actual source-image steps | All methods updated the backbone; matched initial weights and augmented batches; no target loader | `verification/real_data_cpu_smoke.json` |
| Resume | Bitwise identical CPU model and Adam states for ERM/RSC/cue | `tests/test_officehome_training.py` |
| Matrix and reports | 36 training runs before any target access; complete-run reuse; matched artifacts and correct domain columns | Tests |
| Fail-fast driver | Failed GPU preflight prevents later training; unavailable-CUDA driver created no benchmark run directory | Driver test and actual CPU-host invocation |
| Source and static audit | Passed; 14 new Python files; zero inline comments | `verification/audit.json` |
| CUDA fit and CUDA pretrained steps | Unavailable on this host; not measured | `verification/cuda_memory.json` |

The documented `run.py prepare` command was executed against this canonical checkout, including the complete combined test suite, the real original dataset, and CPU source steps. Preparation uses the same dataset ID `405a956255b0608e8a74a8514384f3fe3e0914a3a8e33c069598de5539f0abbe` after relocation.

The source image collection digest is `a1530596c2be0d4553a7188c1248fd1ef203755d65c7055fc8053763bd242249`. Archive hashes are measured source checksums, not an author-published guarantee.

No real Office-Home target accuracy, full matrix outcome, GPU runtime, or method advantage was produced here. On the GPU server, `run.py train` requires a successful full-batch memory profile and ten pretrained real-source CUDA updates per arm before launching the 36-run matrix. Finish the single-GPU PACS experiment first.
