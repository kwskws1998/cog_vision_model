# Classification implementation review

This file describes review scope. Actual pass counts and source fingerprints are written by the tests and audit under ignored `verification/` files.

## Uploaded ERM archive audit, 2026-10-23

The supplied `main.zip` SHA-256 is `c631422eed49b6131c2a2c73e14fb46940ac9a53487ff18094fe13fdc6189cb2`. It contains four ERM run directories, three completed training runs, and one incomplete training snapshot. It contains no final-test predictions, final-test metrics, or whole-study seal. Its contracts identify BF16 training and FP32 evaluation, not FP16. These are observations about this ZIP snapshot, not the current remote process.

| Saved ERM run | Last saved epoch | Accepted updates | Online training accuracy | Internal validation accuracy |
|---|---:|---:|---:|---:|
| CIFAR-100 seed 0 | 200 | 8,800 | 99.9822% | 73.42% (5,000 images) |
| CIFAR-100 seed 1 | 200 | 8,800 | 99.9867% | 73.48% (5,000 images) |
| Tiny ImageNet 64px seed 0 | 200 | 17,600 | 99.9756% | 61.89% (10,000 images) |
| Tiny ImageNet 64px seed 1 | 25 | 2,200 | 74.01% | 46.18% (10,000 images) |

The incomplete Tiny seed 1 runtime snapshot says epoch 26, while its last durable checkpoint/history end at epoch 25. It has no final checkpoint, summary, or completion seal. The earlier CUE report contains official final-test accuracies: CIFAR 73.06%/73.29% and Tiny 62.25%/63.04%. Its confusion matrices independently reproduce those accuracies and the reported seed means/SDs. **Subtracting these CUE final-test scores from the ERM internal-validation scores would not estimate the intervention's effect.** No CUE training checkpoint or history was supplied in this ZIP, so actual historical CUE gate trajectories and ERM/CUE checkpoint duplication cannot be checked from it.

Committed Git blobs reproduce both recorded source identities: ERM `4fb74cfa…` belongs to `6b57dc0a1b57db941a75810316907d076cf11e3f`; CUE `1462200e…` belongs to `1c06f8d43724987817348fc869bd02f6fb4ee92f`. Between those commits, the model, objective, and metric mathematics are unchanged; optional Tiny32 support preserves the original Tiny64 and CIFAR transforms. The configurations match apart from method/seed and the newly explicit default Tiny size. Current versus historical ERM/CUE execution also gives identical shared initialization, loss, and parameter gradients on a matched diagnostic input.

Artifact integrity checks loaded all seven checkpoints using `weights_only=True`. All model tensors are finite, all four run model hashes differ, all canonical contract identities match, and all nine completion file hashes match. External histories equal checkpoint-embedded histories, and every completed final state equals its corresponding last state. Every one of 625 saved epoch records has the correct sample/update counts and requested learning rate; no AMP overflow retries were recorded. The recorded LR is 0.16 in epoch 1, 0.8 in epochs 5–100, 0.08 in epochs 101–150, and 0.008 thereafter.

Independent pipeline checks verified ordinary fully trained ERM cross-entropy, all-parameter SGD, encoder-only autocast, FP32 readout/evaluation, restored training mode every epoch, evaluation without BatchNorm updates, and prediction-derived metrics. The local real dataset identities match the archive contracts. Independently reconstructed split membership is identical, class balanced, and disjoint: CIFAR 45k/5k/10k and Tiny 90k/10k/10k. Six targeted vanilla-ResNet, all-gradient, metric, augmentation, and BatchNorm regressions passed.

For all four uploaded checkpoint states, an independently constructed torchvision ResNet, without importing the project model or loss, replayed every internal-validation image on CPU FP32: 30,000 image evaluations in total. The accuracies exactly reproduce **73.42%, 73.48%, 61.89%, and 46.18%**, with NLL differences below `5e-7` from the recorded CUDA FP32 evaluations. The last value belongs to the incomplete epoch-25 Tiny run. No official final-test images were opened. This supports the saved validation calculations and the observed generalization gap; it does not establish a CUE benefit.

The CUE derivative was rederived independently in NumPy through features, shared classifier weights, and bias at 2/7/100/200 classes and unequal region areas. Maximum absolute error was `1.39e-16`. The actual `optimize_batch` path on identical initial weights and 16 actual CIFAR training images produces distinct ERM/CUE updates: PostAct L2 parameter difference `0.01854` across all 62 parameter tensors at LR 0.1; PreAct `0.009096` across 56 tensors. The all-joint endpoint matches ERM gradients exactly when using the same joint-logit graph; the direct-GAP production path differs by at most `1.23e-7` in updated PostAct parameters because of FP32 averaging order. No disabled CUE branch or gradient-cancellation error was found.

The remaining structural concern is the intended correction itself: `delta_k = (a_k/B) * (1-gamma_k) * (p_local_k-p_joint)`. Agreement makes the correction small, while strong disagreement drives the gate toward the joint/ERM update and suppresses it again. Its default maximum cue-logit norm is approximately `0.0550534*a_k/B`; this is an absolute bound, not a 5.5% bound relative to ERM gradients. On one deterministic batch of 32 real training images, applying CUE counterfactually to the supplied ERM final weights changes the whole-parameter gradient by 11.57%/6.36% for CIFAR seeds 0/1 and 22.32% for Tiny seed 0. These CPU FP32, batch-32 diagnostics are **not measurements of the historical CUE training trajectory**, and BatchNorm conditions differ from batch 1024.

The evidence supports working ERM/CUE implementations but does not show a positive CUE treatment effect. Similar accuracy can reflect an ineffective or weak regularizer rather than a software defect. A matched final-test ERM comparison is still missing from this attachment, and a fixed-consistency control remains necessary to attribute any improvement specifically to adaptive gates. No training objective or hyperparameter was changed in response to similarity alone. Reproduction scripts and machine-readable audit evidence are retained locally under `.cache/erm_archive_review_20261023/`; original uploaded files and historical contracts are unchanged.

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
