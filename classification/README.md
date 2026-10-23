# Controlled classification experiments

This directory implements CIFAR-100, CIFAR-100-C, native-64px Tiny ImageNet-200, and the declared MNIST-CIFAR cue-conflict experiment. It shares the project environment and core operators with PACS and Office-Home. Each study records its own resolved recipe and source identity.

The main comparison is ERM, uniform non-target label smoothing, fixed global-to-local self-distillation, Zipf's Label Smoothing dense2, and CUE. These are training interventions on a common small-image ResNet-18. Final inference is global average pooling and the shared final classifier. CUE does not adapt or choose cues during testing.

## Server execution

Run from the project directory, for example `/workspace/cog_vision_model`. Install the shared Conda environment using the [root setup instructions](../README.md#setup). The commands below use Bash and stop if any stage or a piped training process fails.

Prepare the data once:

```bash
(
set -e
set -o pipefail

cd /workspace/cog_vision_model
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/.conda"

python -u classification/run.py prepare \
  --data-root data/classification
)
```

Preparation runs the combined tests, downloads/verifies all required data, and checks CPU updates on real training images. It does not compute final-test predictions. Downloads resume interrupted transfers and verify complete release checksums before extraction.

The editable experiment inputs are in [`configs/experiment.json`](configs/experiment.json). `clvision/settings.py` validates and resolves them; `clvision/workflow.py` executes the resolved study; `clvision/engine.py` implements training and evaluation. The study includes both natural datasets, all five main methods, and the six mechanism methods. Edit the JSON or the explicit arguments below to change its settings.

Preview the exact recipes and run count without loading datasets or starting training:

```bash
python classification/run.py train \
  --config classification/configs/experiment.json \
  --seeds 0 1 \
  --epochs 200 \
  --milestones 100 150 \
  --mechanism-epochs 50 \
  --mechanism-milestones 25 40 \
  --batch-size 128 \
  --eval-batch-size 128 \
  --learning-rate 0.1 \
  --dry-run
```

Launch the complete study with explicit inputs and a separate timestamped log:

```bash
(
set -e
set -o pipefail

cd /workspace/cog_vision_model
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/.conda"
mkdir -p results

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="classification_trainfp16_evalfp32_b128_seeds0_1_${STAMP}"

python -u classification/run.py train \
  --config classification/configs/experiment.json \
  --data-root data/classification \
  --runs-dir "runs/${RUN_NAME}" \
  --seeds 0 1 \
  --development-seed 2027 \
  --epochs 200 \
  --milestones 100 150 \
  --mechanism-epochs 50 \
  --mechanism-milestones 25 40 \
  --batch-size 128 \
  --eval-batch-size 128 \
  --learning-rate 0.1 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --lr-decay 0.1 \
  --precision fp16 \
  --eval-precision fp32 \
  --device cuda \
  --workers 4 \
  --max-vram-gib 20 \
  2>&1 | tee "results/${RUN_NAME}.log"
)
```

`--seeds` sets the main and mechanism seed lists together. `--main-seeds` and `--mechanism-seeds` override their respective lists if supplied; duplicate, empty, or out-of-range lists are rejected. The default remains seeds 0 and 1, with development seed 2027: 18 validation-candidate runs, 20 natural-image main runs, and 12 mechanism runs, totaling 50 training runs. Three seeds (`--seeds 0 1 2`) produce 66 runs with the default candidate grids. The preview calculates counts from your actual settings.

Precedence is built-in defaults, JSON common training settings, JSON dataset-family schedules, JSON dataset overrides, then explicit CLI flags. Natural-image `--epochs`/`--milestones` and mechanism `--mechanism-epochs`/`--mechanism-milestones` are separate. Supply compatible milestones when changing epochs; `--milestones` with no values disables the natural-image milestone schedule. Validation grids and dataset-specific coefficients can be edited in JSON; each dataset keeps an equal positive candidate budget across the three tuned methods.

The same resolved recipes drive CUDA preflight and training. Preflight checks every intervention using actual data, the requested training/evaluation batch sizes and precision, and ten accepted optimizer updates. A failed preflight stops the study. The workflow then develops on internal validation, freezes choices, trains all main/mechanism runs, seals the complete study, and evaluates the final checkpoints, including CIFAR-100-C. Results are written to the selected run directory as `comparison.json` and `comparison.md`.

Training precision (`--precision`, default `fp16`) and evaluation precision (`--eval-precision`, default `fp32`) are independent, recorded settings. FP32 evaluation applies consistently to preflight validation, every epoch's validation, clean tests, mechanism diagnostics, and corruption tests. Early training with BatchNorm running statistics can produce validation activations outside FP16's range even when training updates are finite. Evaluation therefore defaults to FP32, without an automatic per-batch precision fallback. Preflight measures its actual throughput and memory cost. Explicit `fp16` and `bf16` evaluation remain available; a precision change requires a new study directory.

Preflight failures retain `status=failed`, the dataset, method, stage, accepted update count, resolved recipe, and exception in `classification/verification/<dataset>_cuda.json`. Evaluation reports nonfinite logits separately from invalid label IDs and includes input/model-state diagnostics. Corrupted floating-point model buffers stop training before an optimizer update is accepted. Partially consumed preflight data loaders are shut down on failure so their worker/pin-memory errors do not obscure the original error. These checks do not clamp or replace invalid predictions.

Timing is automatic. Preflight prints training/evaluation throughput for each dataset and method. Before the first full training run, `[study]` prints a provisional whole-study ETA based on those measurements. It includes development, all main/mechanism epochs, per-epoch validation, clean tests, mechanism panel diagnostics, and all 750 CIFAR-100-C checkpoint/condition evaluations under the default two-seed scope. Actual epoch times, including checkpoint writes, replace provisional rates as training proceeds; measured final evaluation times refine that stage's estimate. Startup, final sealing, report generation, and downtime are not predicted. An unavailable measurement is shown as `estimating`, never an invented duration.

Every completed epoch prints the active run's elapsed/remaining time and a whole-study line with completed runs, clean evaluations, corruption cells, session elapsed time, remaining time, and estimated finish in KST (`+09:00`). Long training epochs also print batch progress approximately every 30 seconds, after a batch completes. The current snapshot is saved atomically to `runs/<RUN_NAME>/progress.json`; completed jobs/epochs/corruption cells are recovered when resuming. Once preflight succeeds and the progress tracker starts, failures are recorded as `status=failed`; successful completion records the actual session duration and `status=complete`. A real RTX 5090 duration becomes available after its preflight; CPU smoke timing is not a GPU estimate.

To resume, reuse the original `--runs-dir` and identical settings, and use `tee -a` to append to its log. Do not generate a new timestamp for a resume. Completed runs are verified and reused; incomplete runs resume from the last complete epoch. A changed seed list, recipe, dataset, or executable source requires a new run directory. `plan.json` records the resolved settings and recipes before training; existing contracts are not rewritten to accept changes. The JSON plan and per-run contracts are the authoritative record for a customized experiment.

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

The default development grid has three final-epoch validation candidates per tunable method, using training seed2027. CUE thresholds are0.05/0.15/0.30 at temperature0.05 and joint floor0.5; fixed KL weights are0.1/0.25/0.5; Zipf KL weights are0.03/0.1/0.3 on CIFAR and0.3/1/3 on Tiny. Validation selection prefers accuracy, then lower NLL, then registered candidate order. Candidate completion hashes are bound into the frozen selection and main-run contracts.

## Evidence and limitations

The primary question is whether the adaptive method improves on fixed consistency on both natural datasets. ERM and label smoothing test practical value; Zipf is the selected external comparator. All seeds and methods are reported. With only two pilot seeds, bootstrap intervals are descriptive and unstable; the code does not claim statistical significance or2026 state of the art. A corruption-only gain with lower clean accuracy is a tradeoff. A mechanism-only gain does not establish general image classification utility.

The completed PACS result remains a negative ERM comparison. Existing Office-Home training remains supplementary DG evidence. This code does not implement a new ViT benchmark or prove the cognitive interpretation of final feature-map regions.

See `REVIEW.md` for implementation checks, `SOURCES.md` for exact paper excerpts, and generated `verification/` reports for the source-bound outcomes actually measured on the executing machine.

The default settings retain the two-seed pilot. Existing study plans and checkpoints cannot be resumed across changed source or scope. Finish or stop an existing training process before updating code; keep its original results as a separate study.

The audit retains historical PACS/Office-Home hash comparisons as provenance information; current verification still requires matching current source hashes and successful outcomes. Because the configurable workflow also updates shared setup and CLI modules, distribute the full project using `python bundle.py --output /absolute/path/to/cog_vision_model.zip`. `classification/bundle.py` produces the same complete source layout after requiring a current passing classification audit. Neither package includes datasets, environments, caches, or run checkpoints.
