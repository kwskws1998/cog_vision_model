# Classification implementation review

This file describes review scope. Actual pass counts and source fingerprints are written by the tests and audit under ignored `verification/` files.

## Data and experimental decisions

- Original CIFAR fine classes, all200 Tiny classes, native dimensions, fixed stratified source validation, and content hashes are explicit.
- Digit and vehicle identities are separated before pairing; only vehicle labels define the mechanism outcome. Counterfactual tests use identical vehicle images under both digits.
- Only15 registered corruptions at5 levels enter the corruption average. Clean checkpoint weights and normalization remain fixed.
- Fixed-final-epoch checkpoints prevent target-derived selection. Development uses exactly three candidates per tunable intervention and freezes their scores and completion identities.
- All planned natural and mechanism training finishes before final evaluation. Production scope is50 main plus18 development plus18 mechanism runs.

## Numerical operators and optimization

- The CUE KL expression is checked against the original surrogate's gradients through both shared features and head parameters, with multiple class counts and unequal cue areas.
- Zipf ranks and KL are compared with an independent reference, including zero-count ties, target exclusion, binary labels, finite differences, and scalar-shift invariance.
- Zipf's dense2 auxiliary head and CE are retained and counted. Its auxiliary initialization cannot change shared weights or CPU/CUDA training randomness.
- All methods share the encoder, optimizer, augmentation, sample order, normalization, inference rule, and paired training seeds.
- Exactly accepted SGD updates are counted. AMP replay restores RNGs and BatchNorm buffers; persistent/non-scaled nonfinite gradients and nonfinite forward losses are fatal.
- Exact CPU resume checks model, SGD momentum buffers, scaler state, RNGs, learning-rate boundaries, and completed update counts for every main method.

## Artifact and runtime checks

- Source, data, configuration, initial shared weights, class order, runtime, development selection, and experiment phase are bound into each contract.
- Final checkpoint, history, summary, and completion hashes are checked before final evaluation and report generation.
- Reports recompute clean metrics from saved predictions and reject changed sample identities, canonical targets, or ordering. Corruption summaries require the exact 75-condition grid and full sample counts, and expose paired seed differences plus corruption/severity aggregates.
- Completed runs are verified/reused. Training does not continue after final-test access. Corruption evaluation resumes completed cells with matching data and checkpoint identities.
- A process lock excludes concurrent classification jobs before CUDA preflight. Epoch checkpoints recover from process interruption without reusing a partial SGD state.
- The real-data preflight measures every intervention with full batch/evaluation shapes, initialized momentum, and at least ten accepted CUDA updates. Failure stops before benchmark study creation.
- Full confusion matrices are retained in final prediction metrics, while epoch histories keep compact validation metrics to avoid unnecessary checkpoint growth.

## Limits

CPU tests and synthetic studies verify code paths, not efficacy or target-GPU fit. Full-batch CUDA memory, speed, and AMP behavior must be measured on the user's server. This development host has no CUDA device.

The common torchvision post-activation backbone differs from Zipf's original small-image pre-activation backbone; native64px Tiny differs from its published32px recipe. The comparison is an explicit matched operator port. The module is training-only and no ViT efficacy experiment is implemented here.

## Measured local verification, 2026-10-23

The final combined run completed 125 tests with zero failures, errors, or skips in the existing Conda environment. The complete synthetic study passed, reused completed checkpoints exactly, and rejected reordered prediction identities. All five main methods accepted real CPU training updates on CIFAR-100, Tiny ImageNet, and Dominoes, with identical shared initialization and first augmented batches within each dataset. CPU checks are explicitly marked as verification only.

A standalone CUDA profile returned exit code 2 with `measured=false` and `passes_budget=false` because this Mac has no CUDA device. It did not substitute CPU evidence for GPU fit. The server's full-batch preflight is still required.

The authoritative executable fingerprint and generated outcomes are in the packaged verification JSON files. No benchmark accuracy or superiority conclusion follows from these checks.
