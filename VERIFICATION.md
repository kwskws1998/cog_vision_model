# Verified implementation status

The current source fingerprint is `8012ca8004e2a2f625f2e7d861c8f88cd9099073fbad26c7e6b43ca49df31258`. Reports were regenerated against these exact executable sources. This is implementation evidence, not a completed domain-generalization experiment.

| Check | Actual outcome | Evidence |
|---|---|---|
| Conda CPU environment | Created successfully; actual module locations, build, and dependencies verified | `verification/conda_cpu_setup.json` |
| Numerical and protocol tests | 46 passed, zero failures/errors/skips | `verification/unit_tests.json` |
| RSC mask reference | Independent NumPy percentile calculation and exact-tie cases passed | `tests/test_operators.py` |
| Cue gradient | Analytical per-cue derivative and all-joint ERM reduction passed | `tests/test_operators.py` |
| Spatial/token readout | Odd-region area preservation and CNN/ViT-token equivalence passed | `tests/test_operators.py` |
| Resume | Bitwise identical CPU model and optimizer states for all three methods | `tests/test_protocol.py` |
| Data workers | Workers 0 and 2 yielded identical images, labels, and augmentations | `tests/test_protocol.py` |
| Protocol | Target-loader exclusion during training, final seal, and changed-checkpoint rejection passed | `tests/test_protocol.py` |
| Result reporting | Saved-prediction metric recomputation, paired aggregation, and recipe checks passed | `tests/test_protocol.py` |
| Full pipeline | Real ResNet-18 trained, selected, saved, and evaluated on artificial images for every arm | `verification/cpu_smoke.json` |
| Pretrained 224px | Finite optimizer step and actual backbone update for every arm | `verification/pretrained_cpu_smoke.json` |
| Real PACS package | All 9991 image files verified, fully RGB-decoded, and hashed; all 12 official split blobs matched | `verification/dataset_preparation.json` |
| Real PACS source-image steps | All arms passed; initial weights and augmented training batches matched; no target loader opened | `verification/real_data_cpu_smoke.json` |
| Source audit | 19 Python files parsed; recipe/split checks passed; all reports match the current source | `verification/audit.json` |
| Static checks | Ruff checks and formatting passed for all project Python files | `verification/static_checks.json` |
| Cancellation regression | Original gradient error 0.3156648874 reduced to 0 in the controlled FP32 case | `verification/review_regressions.json` |
| CUDA memory | Unavailable; not measured; profiler exited 2 and wrote the reason | `verification/cuda_memory.json` |

The fresh environment used macOS 15.5 ARM64, Conda Python 3.11.17, torch 2.8.0, torchvision 0.23.0, NumPy 2.0.2, Pillow 11.3.0, gdown 5.2.0, and filelock 3.19.1. The previous environment was removed; the active project environment is `.conda`. Native shared-memory permission was enabled for the multi-worker test. Other verification ran in the workspace sandbox. The CUDA 12.6 installation path and AMP execution must still be exercised on the target Linux GPU.

The pretrained 224px smoke uses synthetic inputs with three images per batch. The real-PACS smoke uses three real source images per training step and three source-validation images from each source domain. Both execute one optimizer step per method. Neither is the batch-48/5000-step benchmark. There is no completed PACS target-accuracy comparison, estimated runtime for the full matrix, measured 24 GB fit, or demonstrated method superiority.

The downloaded Dassl PACS archive has SHA-256 `0dc9d0176fa27c9b4504e7c2e962aebe6a79ed0c1819b84148786e590f87e102`. Its prepared dataset ID is `7efbd50341b5cc38fe816aa3cd9ffa369dabb9f0e4d8654d1ee8bb0b54180bd6`. The DomainBed Google Drive link was unavailable during verification; the documented Dassl fallback succeeded. Both download sources are recorded in SOURCES.md. The portable ZIP excludes the images and pretrained weights; prepare/download them on the training machine.

Before the full experiment, run `experiment.py profile` on the actual otherwise free 24 GB-class GPU. A passing result must cover all three arms at the unchanged batch-48/224px recipe and the evaluation batch of 64, including materialized Adam state. The code enforces a peak PyTorch reserved-memory budget of 20 GiB and does not silently resize the experiment.

The full review and repaired findings are in [REVIEW.md](REVIEW.md). Conda setup exports are stored in `verification/conda_cpu_explicit.txt` and `verification/cpu_environment_freeze.txt`; the native CPU export is platform specific. All numerical/protocol tests and all three smoke paths were rerun through the documented Conda command. The exact source fingerprint above is checked by the final audit.
