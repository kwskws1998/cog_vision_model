# Classification implementation review

This file describes review scope. Actual pass counts and source fingerprints are written by the tests and audit under ignored `verification/` files.

## Data and experimental decisions

- Original CIFAR fine classes, all200 Tiny classes, native dimensions, fixed stratified source validation, and content hashes are explicit.
- Digit and vehicle identities are separated before pairing; only vehicle labels define the mechanism outcome. Counterfactual tests use identical vehicle images under both digits.
- Only15 registered corruptions at5 levels enter the corruption average. Clean checkpoint weights and normalization remain fixed.
- Fixed-final-epoch checkpoints prevent target-derived selection. Development uses exactly three candidates per tunable intervention and freezes their scores and completion identities.
- All planned natural and mechanism training finishes before final evaluation. The requested pilot scope is 20 main plus 18 development plus 12 mechanism runs, using paired seeds 0 and 1 for main/mechanism training.

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

The default torchvision post-activation backbone differs from Zipf's original small-image pre-activation backbone; the selectable PreAct implementation follows the pinned author block/stem and initialization. This choice alone does not reproduce its complete protocol. Native64px Tiny also differs from its published32px recipe. The comparison is an explicit matched operator port. The module is training-only and no ViT efficacy experiment is implemented here.

## Selectable backbone verification, 2026-10-23

The combined Conda suite passed 251 tests with zero failures, errors, or skips after adding `activation=postact|preact`. Independent tests load the pinned Zipf reference implementation and compare PreAct shared initial tensors, logits, all shared parameter/input gradients, and BatchNorm updates. A preserved old constructor checks exact PostAct state keys, RNG, initial tensors, and predictions when the option is omitted or explicitly `postact`.

Additional checks cover all eight training interventions, 32px/64px/rectangular maps, signed features and unequal-area CUE readout gradients, auxiliary initialization isolation, both CLI entrypoints and JSON precedence, exact PreAct checkpoint resume/final reconstruction, real synthetic ERM/CUE workflows, and accurate backbone labels in saved reports. Changed activation invalidates a CUDA profile, checkpoint resume, or frozen study plan; a full-study report cannot pool both architectures under one dataset label. These are implementation checks, not accuracy or GPU throughput measurements.

The separate `classification/run.py smoke --activation preact` also passed: 34 reduced synthetic training runs, 136 accepted optimizer steps, and 16 clean final evaluations across CIFAR-100, Tiny ImageNet, and Dominoes. It reused completed artifacts exactly and rejected deliberately reordered predictions. The expected rejection is logged as a failed tampered study; the smoke command itself returned success after checking that rejection.

## Original implementation verification, 2026-10-23

Before the two-seed amendment, the combined run completed 125 tests with zero failures, errors, or skips in the existing Conda environment. The complete synthetic study passed, reused completed checkpoints exactly, and rejected reordered prediction identities. All five main methods accepted real CPU training updates on CIFAR-100, Tiny ImageNet, and Dominoes, with identical shared initialization and first augmented batches within each dataset. CPU checks are explicitly marked as verification only.

A standalone CUDA profile returned exit code 2 with `measured=false` and `passes_budget=false` because this Mac has no CUDA device. It did not substitute CPU evidence for GPU fit. The server's full-batch preflight is still required.

The authoritative executable fingerprint and generated outcomes are in the packaged verification JSON files. No benchmark accuracy or superiority conclusion follows from these checks.

## Two-seed amendment

The main and mechanism seed lists are both exactly `[0, 1]`; development retains seed 2027 and all nine candidates per natural dataset. The production-scope regression checks 20 main and 12 mechanism runs, and rejects an existing plan containing the previous seed lists. The amended default output directory is `runs/classification_2seeds_v1`. Fresh outcomes are recorded by the current-source test report.

## ERM log and CUE implementation audit, 2026-10-23

The supplied log fragment contains CIFAR-100 ERM seed 1 epochs 73–200 and Tiny ImageNet ERM seed 0 epochs 1–107. It contains no final-test summary. Its old console format predates the Trainer-style logging change; the exact remote source fingerprint and checkpoints were not included. The inspected training/evaluation mathematics is unchanged between `6b57dc0` and `7763220`.

| Internal-validation result | Epoch 100 | Epoch 101 |
|---|---:|---:|
| CIFAR-100 ERM seed 1 accuracy | 57.36% | 70.64% |
| Tiny ImageNet ERM seed 0 accuracy | 44.18% | 59.44% |
| Configured learning rate | 0.8 | 0.08 |

The scheduler explicitly changes LR after epoch 100, and again after epoch 150. The sharp improvements align with this schedule. This supports investigating the aggressive LR regime, but does not prove which portion of the gap comes from LR, generalization, BatchNorm statistics, or GPU precision. The old `loss` field was training loss: CIFAR seed 1 ends with training CE 0.0047 and internal-validation accuracy 73.48%. That is not validation CE 0.0047. Comparing this 73.48% against the historical CUE final-test mean 73.18% would mix splits, seeds, and metrics.

The batch-1024 recipe has 8,800 CIFAR optimizer updates over 200 epochs, versus 70,400 at batch 128. Linear LR scaling to 0.8 did not establish accuracy equivalence. Preserve the current matched baseline run; any lower-LR study must use a separate frozen plan, validation-only selection, and the same recipe across compared methods. Existing ablation settings support LR candidates 0.1, 0.4, and 0.8. Do not change LR or recalibrate BatchNorm using validation/test data halfway through this experiment.

Independent checks with real local CIFAR images and a separately reconstructed torchvision small-image ResNet-18 found identical ERM logits, CE, and all parameter gradients. FP32 evaluation matched manual argmax accuracy, was invariant to evaluation batch partitioning, and left all BatchNorm buffers unchanged. Validation inputs were independent of epoch/seed. All local Tiny ImageNet image records were integrity-checked: 90,000 training, 10,000 internal-validation, and 10,000 final-evaluation images, with 450/50/50 examples per class and no training/internal-validation overlap. Both Tiny resolutions produced deterministic validation tensors. These are local data/code checks, not validation of unseen remote checkpoints.

CUE's independent NumPy derivative checks agree with the documented categorical operator to approximately 1e-16 in FP64, including unequal cue areas, shared features, classifier weights/bias, and all-joint/all-marginal limits. Teacher and gates are detached correctly. No mathematical CUE or ERM evaluation defect was found in the inspected paths. See [the classification gradient derivation](../METHOD.md#classification-implementation).

Paper fidelity is a separate issue. [Mi and Summerfield, Methods Model 4 and Results](https://www.nature.com/articles/s41562-026-02452-1) state, verbatim:

> we adopted the hybrid s.d. model

Their implemented PE strategy is trial-common, using distance across presented-cue error sets. Our per-region categorical probability distance, smooth gate, and 0.5 joint floor are design modifications. Their later analyses use cue-weight variability; this code is PE-inspired. It is therefore a proposed vision adaptation, not a reproduction of the fitted human model.

The small-image ResNet's theoretical final receptive field is 109px, computed from its actual convolution/stride sequence. Thus final-map regions at 32px and 64px inputs are not isolated input patches; this does not assert identical effective receptive fields. Demonstrating a CUE gain still requires paired final-test ERM results and a fixed-consistency control to isolate adaptive gating. The supplied ERM fragment provides no evidence that CUE gates vary meaningfully.

One telemetry defect was identified and fixed: resetting CUDA peaks without releasing inactive cache retained the previous preflight's reservation baseline. Each run now releases inactive cache before model construction, and logs both peak allocated and reserved memory. The PyTorch 2.8 allocator's [peak-reset implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/c10/core/Allocator.h#L338-L340) resets peaks to current counters. Existing measurements are retained; the cache correction does not retroactively recover per-run tensor memory. Epoch logs also name training metrics explicitly and expose CUE loss components and within-image gate variation. Neither the objective nor preset hyperparameters changed.

Resolving the remaining run-specific hypothesis requires its saved history and checkpoint. Compare clean training images and internal validation with the same checkpoint in FP32 eval mode. If testing BatchNorm recalibration, use a copy and training data only, label it as a separate diagnostic, and retain the original reported weights/results. No such remote-checkpoint or RTX 5090 diagnostic was performed in this audit.

After the diagnostic fixes and independent regressions, `classification/run.py check` passed all 230 tests in the local Conda environment. This includes the new vanilla-ResNet evaluation checks, independent NumPy CUE derivatives, gate-statistic semantics, and memory-reset ordering. CUDA cache behavior was checked against the allocator implementation and simulated counters, not measured on a local GPU. Ruff and whitespace checks passed.
