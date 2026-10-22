# Prespecified comparison

## Task and controls

PACS has four domains and seven classes. The manifest contains 2,048 art paintings, 2,344 cartoons, 1,670 photos, and 3,929 sketches. Each target run trains on official training subsets of the other three domains and selects a checkpoint using only their official cross-validation subsets. The held-out domain's complete official test list supplies the final target evaluation.

| Domain | Train when source | Validation when source | Test when target |
|---|---:|---:|---:|
| art_painting | 1840 | 208 | 2048 |
| cartoon | 2107 | 237 | 2344 |
| photo | 1499 | 171 | 1670 |
| sketch | 3531 | 398 | 3929 |

For each domain, training and cross-validation images are disjoint and their union equals its target test list. The class order is dog, elephant, giraffe, guitar, horse, house, person. Original 1-based labels are converted once to 0-based labels. Source train/cross-validation membership is fixed by the pinned official files, not randomly regenerated.

| Setting | Shared recipe |
|---|---|
| Backbone | torchvision ImageNet1K-v1 ResNet-18 |
| Trainable weights | Entire encoder, BN affine weights, one 512→7 classifier |
| BN running statistics | Frozen to pretrained statistics in every arm |
| Training length | 5000 optimizer steps, constant LR |
| Optimizer | Adam, lr=0.00005, weight decay=0 |
| Training batch | 48: 16 independently sampled images per source domain |
| Sampling | With replacement, deterministic by seed/step/domain |
| Training augmentation | RandomResizedCrop(224, scale 0.7–1.0, default ratio), flip p=0.5, ColorJitter(0.3 each), grayscale p=0.1 |
| Evaluation | Resize shortest side to 256, center crop 224 |
| Normalization | ImageNet mean/std |
| Precision | FP16 CUDA encoder, FP32 classifier/loss/gates, GradScaler |
| TF32 | Disabled for cuDNN and matrix multiplication |
| Validation | Every 200 steps, all three source cross-validation subsets |
| Checkpoint selection | Highest equal-domain mean accuracy, then lowest equal-domain mean NLL, then earliest step |
| RSC | Feature drop 1/3, batch drop 1/3; pooled 512-D features |
| Cue routing | 2×2 area-preserving grid, tau=.15, temperature=.05, joint floor=.5 |
| Primary seeds | 0, 1, 2 |
| GPU memory budget | Peak PyTorch reserved ≤20 GiB, checked throughout training |

Class balancing is not applied; domain balancing is identical across arms. The grid affects only cue routing's training loss. ERM and RSC use pooled features during training; all three use identical pooled features and the shared classifier at evaluation. The all-joint cue loss has ERM's analytical gradient, verified numerically through the shared head and backbone interface.

## Baseline scope

RSC is a **DomainBed-style pooled-feature adaptation** pinned to the repository revision in SOURCES.md. The original paper's stronger spatial-plus-channel implementation and published SGD recipe are not reproduced by this package. Do not compare this implementation's score directly to a published table and infer an improvement. The common Adam/AMP/augmentation/source-selection recipe is an explicit experimental choice. Domain labels are used for balanced sampling and source validation by every arm; the RSC operator itself does not receive domain labels.

The mask uses gradients of correct-class raw logits with respect to pooled features. It does not use absolute gradients or cross-entropy gradients. Values at or above the feature percentile are muted. Samples whose probability drop is below the batch percentile retain every feature. All other samples use the feature mask. Both cutoffs use strict `<` comparisons, preserving tie behavior. Thresholds are FP32 during the benchmark, with a double-precision linear quantile calculation matching the NumPy percentile definition. Final evaluation never applies masks.

## Integrity and data access

Preparation reads all images to check integrity and compute SHA-256 hashes. Training checks the complete manifest again for reproducibility, but constructs loaders only for source training and source validation. It never computes target predictions or metrics. Tests explicitly fail if source selection opens a target or test loader.

`contract.json` binds configuration, image/split identity, code, initial weights, pretraining file identity when provided, runtime, and selection rule. Initial weights must match across the three arms for a target/seed pair; matrix preflight enforces this before any target access, and the report checks again. The matrix plan binds the dataset ID before training. CPU thread count, software versions, precision flags, and available GPU identity are recorded. Exact resume is tested locally on CPU; bitwise CUDA resume remains unverified until exercised on the target environment. Deterministic flags cannot promise equivalence across arbitrary drivers or GPUs.

`last.pt` atomically contains current weights, Adam/scaler state, RNGs, history, and the selected weights. This avoids a crash pairing an old optimizer state with a newer selected model. Recovery restarts from the last completed validation checkpoint. At the planned final step, `best.pt` and the complete summary are committed before the SHA-256 selection seal becomes available. Missing or stale summaries are reconstructed from a completed checkpoint. A kernel-backed writer lock prevents simultaneous writers and is released on process death. Explicit final evaluation checks the seal before opening the target loader and journals target access. Training cannot resume after target access. Repeated final evaluation is journaled; it is not fresh independent evidence.

The complete matrix uses a shared matrix writer lock, verifies/reuses already sealed runs without retraining, and finishes all 36 source-only training runs before its first optional target evaluation. Its plan is saved before running. Use a new run directory for any changed configuration or executable source. Hyperparameter tuning, if later needed, must use only source-domain development procedures; no target-based tuning is implemented here.

An FP16 scaled-gradient overflow does not consume a scheduled optimizer update. GradScaler skips the overflowing attempt, reduces its scale, and retries the same input tensors after restoring model buffers and all training RNGs. At most sixteen scale reductions are allowed for one batch. Thus 5,000 steps means 5,000 completed Adam updates in every arm. Forward nonfiniteness and unrecovered backward failures stop the run. Overflow counts and accepted loss scales are saved with interval diagnostics; scaler state is included in each checkpoint. The same recovery rule applies to all three methods.

## Metrics and interpretation

Primary outcome: top-1 target accuracy, averaged equally over four target domains within each seed, then averaged across seeds. Report sample SD across the three seed-level four-domain means, not standard error over 12 supposedly independent domain/seed scores. Also retain each domain's accuracy and each paired cue-minus-RSC seed difference. Secondary metrics are macro recall across present classes, NLL, multiclass Brier score, 15-bin ECE, and confusion counts. Per-image target logits and exact paths are saved. The report verifies their ordered path/label partition against the training contract, in addition to recomputing every metric.

The report requires a complete, matched real-PACS matrix. Debug runs, generated fixtures, shortened schedules, resized inputs, and random initialization cannot count as primary benchmark evidence. CPU smoke metrics are intentionally omitted from the smoke summary. Source-validation scores alone are not held-out-domain efficacy evidence. No superiority, novelty, or human-mechanism conclusion follows from passing implementation tests.

## Resource boundary

The 24 GB requirement is addressed through a compact backbone, FP16 encoding, no duplicated encoder, no higher-order gradient graph, no extra trainable heads, and sequential execution. RNG capture touches only the active CUDA device, rather than initializing every visible GPU. There are 11,180,103 trainable parameters. The routed head adds zero trainable parameters compared with the matched ERM head. The memory profiler uses actual optimizer steps and evaluation at the unchanged recipe. Its 20 GiB process budget leaves headroom within a 24 GB-class device; CUDA contexts and memory consumed by other processes are not fully represented by PyTorch reserved memory. Actual free memory is also recorded.

The memory guard also runs during every validation and final-test batch. A CUDA OOM or budget violation invalidates that resource check and stops execution. Do not silently reduce one arm's batch or alter one arm's resolution. If the target device requires a smaller recipe, create a new common config, profile all arms again, and rerun the entire comparison. The current host cannot certify the CUDA memory claim.
