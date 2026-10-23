# Controlled classification experiments

This additive directory implements CIFAR-100, CIFAR-100-C, native-64px Tiny ImageNet-200, and the declared MNIST-CIFAR cue-conflict experiment. It preserves every existing PACS and Office-Home executable, recipe, environment file, and checkpoint contract.

The main comparison is ERM, uniform non-target label smoothing, fixed global-to-local self-distillation, Zipf's Label Smoothing dense2, and CUE. These are training interventions on a common small-image ResNet-18. Final inference is global average pooling and the shared final classifier. CUE does not adapt or choose cues during testing.

## Server execution

Use the existing project Conda environment from `/workspace/cog_vision_model`. The new code adds no runtime dependency. Finish any other training on the single GPU first.

```text
git pull --ff-only origin main
conda run --no-capture-output --prefix ./.conda python classification/run.py prepare
```

Preparation first runs the combined legacy and classification tests, then downloads/verifies all required data, then completes CPU updates on real training images. It never computes final-test predictions. Downloads resume interrupted transfers and require complete release checksums before extraction. The corruption archive is about 2.9 GB. CIFAR uses the byte-identical Paddle mirror with the original MD5; Toronto remains a fallback. Tiny's original Stanford endpoint may be unavailable; the downloader then uses the Zenodo mirror only if the complete archive matches the same published MD5. There is no resized-data fallback.

After preparation succeeds:

```text
conda run --no-capture-output --prefix ./.conda python classification/run.py train
```

This command performs real-data CUDA preflight before creating a study: all interventions, batch128, evaluation batch128, SGD momentum state, and ten accepted AMP updates per method. It checks matched initial shared tensors and augmented batches. A failed or unavailable GPU stops the job; CPU verification is never accepted as measured CUDA fit. The peak reserved-memory budget is20GiB, with no silent batch, resolution, or precision reduction.

The full study then executes the following fixed order:

1. CIFAR validation-only development: three candidates each for CUE, fixed consistency, and Zipf; freeze the choices.
2. Eighteen mechanism runs: six interventions, seeds0/1/2. No mechanism final-test result is opened here.
3. Twenty-five CIFAR main runs: five interventions, seeds0/1/2/3/4.
4. Tiny validation-only development and twenty-five Tiny main runs.
5. Seal the complete intended study and evaluate all main/mechanism final checkpoints.
6. Evaluate the same25 CIFAR checkpoints on the75 registered CIFAR-100-C conditions.
7. Write `runs/classification_v1/comparison.json` and `comparison.md`.

There are50 full main runs,18 full validation-candidate runs, and18 shorter mechanism runs. Natural-image runs last200 epochs; mechanism runs last50 epochs. These run counts are not a measured ETA. Each epoch prints validation accuracy, loss, elapsed seconds, and an ETA based on that process's observed speed.

Repeat the same train command after interruption. It rechecks GPU preflight, verifies/reuses completed runs, resumes incomplete runs from the last complete epoch, and resumes corruption evaluation by cell. Mid-epoch work is replayed with identical sample order/augmentation and restored model, SGD momentum, scaler, and RNG state. No model training continues after its final-test access.

Custom locations must be supplied consistently to preparation and training:

```text
conda run --no-capture-output --prefix ./.conda python classification/run.py prepare --data-root /workspace/data/classification
conda run --no-capture-output --prefix ./.conda python classification/run.py train --data-root /workspace/data/classification --runs-dir /workspace/results/classification_v1
```

For a new machine only, use the existing `python3 bootstrap.py --device cuda` before these commands. Do not rebuild an environment currently used by another experiment.

## Standalone checks

```text
conda run --no-capture-output --prefix ./.conda python classification/experiment.py check
conda run --no-capture-output --prefix ./.conda python classification/experiment.py smoke
conda run --no-capture-output --prefix ./.conda python classification/experiment.py prepare --dataset cifar100 --download
conda run --no-capture-output --prefix ./.conda python classification/experiment.py data-smoke --dataset cifar100
conda run --no-capture-output --prefix ./.conda python classification/experiment.py profile --dataset cifar100
```

The complete-study smoke uses explicitly synthetic images, CPU/FP32, one seed, and reduced epochs/batches. It verifies orchestration, complete-run reuse, and rejection of reordered predictions. Its outputs are marked `benchmark_evidence=false` and cannot establish efficacy. On macOS, two-worker tests require permission to create operating-system shared memory.

## Data roles

| Task | Training | Internal validation | Sealed final evaluation |
|---|---:|---:|---:|
| CIFAR-100,32px |45,000|5,000|Official10,000 test images|
| Tiny ImageNet,64px |90,000|10,000|Official10,000 validation images|
| MNIST-CIFAR vehicle components |9,000|1,000|Official2,000 automobile/truck test images|

Natural-image validation uses50 examples per class, selected by fixed SHA-256 ranking of class/index with split seed0. All methods and training seeds share those exact identities. Array files and every labeled Tiny image are hashed; Tiny images are fully RGB-decoded at their native dimensions. Manifest IDs exclude the absolute data root so relocation is supported without changing the data identity. Imported links, archive traversal, duplicate archive names, and changed bytes fail explicitly.

CIFAR-100-C is evaluation only. The fixed15 corruptions and5 severity levels come from the released NumPy archive, with labels checked against the clean test order. The average is raw mean corruption accuracy/error, not normalized ImageNet mCE. There is no corruption fine-tuning, checkpoint selection, or test-time normalization update.

For the mechanism task, RGB-padded MNIST0/1 is placed above CIFAR automobile/truck; the label is always the vehicle. Digit identities are split before pairing. Training/internal validation have95% digit-label agreement. Each held-out vehicle is paired with both digit classes, providing fixed counterfactual predictions weighted into95%,50%, and5% agreement conditions. Report balanced accuracy, all four vehicle-by-digit groups, worst-group accuracy, digit-induced prediction flips, and vehicle-only/digit-only diagnostics. This is our declared variant, not an exact author reproduction.

The mechanism controls are ERM, fixed consistency, CUE, pure marginal supervision, an image-common mean gate, and a fixed permutation `[2,3,0,1]` of `[TL,TR,BL,BR]` gates. The image-common control preserves the per-image total coefficient. The permutation swaps digit-panel and vehicle-panel gates. These deliberately differ from the legacy RMS image gate and shift-one diagnostic. Mechanism gate settings are inherited from CIFAR validation and never retuned on reversed/independent test outcomes.

## Shared recipe and competitor fidelity

All natural methods use torchvision's post-activation ResNet-18, a3x3 stride-one stem, no maxpool, trainable BatchNorm statistics, no pretraining, batch128, and SGD with learning rate0.1, momentum0.9, weight decay0.0001. The rate drops tenfold after epochs100 and150. All use the final epoch200 checkpoint. CIFAR uses padded32px crops and flips; Tiny uses padded64px crops and flips. The only extra Zipf parameters are its training-only penultimate linear classifier.

The Zipf port retains final-plus-penultimate votes, factor-two bilinear spatial upsampling, minimum-tie ranks, exponent1, target exclusion and renormalization, unmasked non-target KL, and auxiliary CE. Auxiliary weights are0.1 for CIFAR and0.5 for Tiny. Its chosen KL weight comes only from the declared validation candidates. The official reference's small-image backbone is pre-activation and its published Tiny experiments resize to32px; our common post-activation/native64px study is a matched reevaluation, not numerical reproduction of the paper. The full pinned reference files, license, and hashes are under `references/`.

CUE uses the current per-cue detached disagreement gate and its exactly gradient-equivalent global-to-local KL expression. The numerical implementation checks gradients through both shared features and classifier parameters. Its scalar logged loss need not equal the legacy surrogate's scalar value. A gain over fixed consistency is required to attribute usefulness to adaptive weighting.

Development is limited to three final-epoch validation candidates per tunable method, using training seed2027. CUE thresholds are0.05/0.15/0.30 at temperature0.05 and joint floor0.5; fixed KL weights are0.1/0.25/0.5; Zipf KL weights are0.03/0.1/0.3 on CIFAR and0.3/1/3 on Tiny. Validation selection prefers accuracy, then lower NLL, then registered candidate order. Candidate completion hashes are bound into the frozen selection and main-run contracts.

## Evidence and limitations

The primary question is whether the adaptive method improves on fixed consistency on both natural datasets. ERM and label smoothing test practical value; Zipf is the selected external comparator. All seeds and methods are reported. Bootstrap intervals over paired seed differences are descriptive; the code does not claim statistical significance or2026 state of the art. A corruption-only gain with lower clean accuracy is a tradeoff. A mechanism-only gain does not establish general image classification utility.

The completed PACS result remains a negative ERM comparison. Existing Office-Home training remains supplementary DG evidence. This code does not implement a new ViT benchmark or prove the cognitive interpretation of final feature-map regions.

See `REVIEW.md` for implementation checks, `SOURCES.md` for exact paper excerpts, and generated `verification/` reports for the source-bound outcomes actually measured on the executing machine.
