# Cue credit routing: a controlled PACS experiment

The selected competitor is **Representation Self-Challenging (RSC)**. The task is **seven-class PACS domain generalization** with an ImageNet-pretrained **ResNet-18**: train on three visual domains and classify images in the fourth. The initial single-target experiment holds out **sketch**. The complete experiment rotates all four target domains.

This package implements the proposed cue-wise marginal/joint gradient routing, a pinned DomainBed-style RSC adaptation, and ERM as the basic control. All three use the same backbone, initial weights, batches, augmentation, optimizer, source-validation selection, and inference rule. RSC is the only named competing algorithm. This is a matched comparison, not a reproduction of RSC's published scores or a claim that RSC is the current state of the art. [Paper excerpts and exact source revisions](SOURCES.md) document the choice.

The implementation targets **one 24 GB-class CUDA GPU**, using 224px images, a global batch of 48, FP16 encoder computation, and FP32 heads and gates. It enforces a **20 GiB PyTorch reservation budget**. CUDA fit must be established by `profile` on the actual GPU; this host has no CUDA device. CPU checks and artificial-image results establish implementation behavior only. No PACS accuracy or method advantage is claimed.

## Setup

Use Conda with Python 3.11. For Linux CUDA, install the pinned CUDA 12.8 PyTorch build through the requirements file. This supports the RTX 5090/Blackwell architecture that the previous CUDA 12.6 build did not support. Run this once from the project directory:

```bash
(
set -e
set -o pipefail

conda env create --prefix ./.conda --file environment.yml
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/.conda"

python -m pip install -r requirements-cuda.txt
python -m pip install --no-deps -e .
python -m pip check
python bootstrap.py --device cuda --verify-only
)
```

For a CPU environment, substitute `requirements-cpu.txt` and `--device cpu`. Install `requirements-dev.txt` in that same environment to run Ruff. `requirements.txt` is the shared dependency list used by the package metadata; the CPU/CUDA files select the appropriate PyTorch build. Environment verification records actual package versions, import locations, the compiled CUDA runtime, and GPU architecture. When CUDA is available it also executes small GPU operations and convolution forward/backward before reporting success; GPU detection alone does not establish kernel compatibility. Full-batch fit is checked separately on the training machine. The official [PyTorch build list](https://pytorch.org/get-started/previous-versions/) provides the pinned 2.8.0/0.23.0 CUDA 12.8 pair, and the [Blackwell support announcement](https://pytorch.org/blog/pytorch-2-7/) documents CUDA 12.8 support.

The optional automated equivalent is `python3 bootstrap.py --device cuda` (add `--dev` for development tools). It creates the dedicated Conda prefix, installs the same requirements, checks dependencies and exports environment records. It rejects the base environment. If `.conda` already exists, activate it and install/verify as needed instead of rerunning `conda env create`.

```bash
conda run --no-capture-output --prefix ./.conda python experiment.py check
conda run --no-capture-output --prefix ./.conda python experiment.py smoke
conda run --no-capture-output --prefix ./.conda python experiment.py pretrained-smoke
conda run --no-capture-output --prefix ./.conda python experiment.py profile
```

For the current ERM/CUE comparison, use the [editable Python experiment](classification/EXPERIMENTS.md). Hyperparameters, seed count, CUE search candidates, stage order, and deadline are supplied by the experiment script rather than fixed inside the workflow. The superseded overnight runner is preserved under [legacy](legacy/README.md).

Classification training now has [Trainer-style step progress, evaluation bars, and structured logs](classification/README.md#trainer-style-progress-logging), enabled by default and configurable without changing the training recipe.

For the next fixed-recipe classification controls, use [ERM-only first, then matched Tiny ImageNet 32px ERM/CUE](classification/README.md#erm-first-then-tiny-imagenet-at-32px). These commands reuse prepared data and the existing Conda environment.

For CIFAR/Tiny ImageNet classification, use the [matched ERM/CUE ablation and bounded tuning workflow](classification/README.md#matched-resnet-ablation-and-bounded-tuning), or the [fixed CUE-only RTX 5090 workflow](classification/README.md#cue-first-on-rtx-5090). Both include explicit seed and hyperparameter flags, a dry-run preview, timestamped logs, and resume instructions.

`profile` measures **ERM, RSC, and cue routing separately**, with the full training batch and the evaluation batch of 64. It materializes Adam state, performs three warmup and three measured optimizer steps per arm, and records peak allocated/reserved memory and GPU properties in `verification/cuda_memory.json`. Validation is included in the memory peak. A passing report is required before describing the configuration as verified on a 24 GB device. It profiles artificial inputs of the correct shape, not classification quality. Other processes can still consume GPU memory; use an otherwise free GPU. No automatic batch reduction changes the experiment.

## Prepare PACS

```text
conda run --no-capture-output --prefix ./.conda python experiment.py prepare --download
```

This tries the PACS archives referenced by DomainBed and Dassl, records the actual successful source, and uses the same pinned split lists for either archive. Google Drive refused the DomainBed download during local verification; availability is external to this package. If both links fail, errors are saved and the command directs you to import an existing archive:

```text
conda run --no-capture-output --prefix ./.conda python experiment.py prepare --archive /absolute/path/to/PACS.zip
```

For already extracted images:

```text
conda run --no-capture-output --prefix ./.conda python experiment.py prepare --data-root /absolute/path/to/PACS
```

The image root must contain `art_painting`, `cartoon`, `photo`, and `sketch`, each with the seven named class folders. ZIP imports also recognize nested `pacs/images/` roots. Preparation verifies and fully decodes all **9,991 images**, hashes their contents, and binds them to the included official train/cross-validation/test lists. Missing or corrupted images stop preparation. Reading image bytes for package integrity does not compute target metrics. Each training invocation checks the manifest and hashes again. The documented Dassl archive successfully passed this complete check locally.

After preparation, verify real-source optimizer steps without launching a benchmark:

```text
conda run --no-capture-output --prefix ./.conda python experiment.py data-smoke
```

This CPU check uses three real source images at 224px per arm and checks unmasked predictions on source-validation images. It verifies matched initial weights and identical augmented batches. It never opens a target loader and does not measure domain generalization accuracy.

Before CUDA training, also exercise the pretrained model on real PACS source batches using the complete benchmark precision and batch sizes:

```text
conda run --no-capture-output --prefix ./.conda python experiment.py data-smoke --device cuda --target art_painting --seed 0 --steps 10
```

This checks the first matrix condition (`art_painting`, seed 0), completes ten actual Adam updates per method at batch 48, checks source-validation predictions at batch 64, and writes `verification/real_data_cuda_smoke.json`. It records FP16 overflow retries and checks that all arms see the same images, augmentations, and initial weights. It does not open a target loader. The random-weight `profile` command measures memory separately and cannot establish pretrained FP16 stability.

FP16 overflow triggers GradScaler backoff and a retry of the same batch, with the original RNG state and model buffers restored. Only a successful optimizer update advances the training step. Recovery is bounded to sixteen backoffs. Nonfinite forward losses, persistent gradient failure, and nonfinite FP32/BF16 gradients remain errors. `amp_overflow_retries` and `amp_scale` are included in training diagnostics.

When updating a checkout that failed with `Nonfinite gradients; this run is invalid, not an implicit skipped step`, preserve its old run directory. Its contract binds the previous source. After applying this fix and passing the CUDA preflight, start a fresh matrix:

```text
conda run --no-capture-output --prefix ./.conda python experiment.py matrix --runs-dir runs/pacs_resnet18_amp_retry_v1 --seeds 0 1 2 --evaluate
```

Do not edit existing contracts to force resume across this source change. Repeating the new command resumes only this new matrix.

ImageNet weights download automatically into `.cache/torch`. Offline runs can use the original torchvision `resnet18-f37072fd.pth` by setting `pretrained_path` in a copied JSON config. Its hash is recorded. Random initialization and reduced-size/short runs require `debug=true` and cannot enter the benchmark report.

## Run the initial sketch comparison

```text
conda run --no-capture-output --prefix ./.conda python experiment.py train --method erm --target sketch --seed 0 --run-dir runs/pilot/sketch_seed0_erm
conda run --no-capture-output --prefix ./.conda python experiment.py train --method rsc --target sketch --seed 0 --run-dir runs/pilot/sketch_seed0_rsc
conda run --no-capture-output --prefix ./.conda python experiment.py train --method cue --target sketch --seed 0 --run-dir runs/pilot/sketch_seed0_cue
```

Each arm runs 5,000 optimizer steps, with source-only validation every 200 steps. A step uses 16 independently sampled images per source domain. BatchNorm running statistics are frozen equally in all arms, while affine parameters remain trainable. Adam uses learning rate 0.00005 and zero weight decay. Full details are in [PROTOCOL.md](PROTOCOL.md).

To resume an interrupted run, repeat its command with `--resume`. Resume verifies the configuration, code, dataset, software/hardware contract, and initial weights. It restores model, optimizer, scaler, selected weights, and RNGs. Stateless sampling preserves the image and augmentation sequence. Recovery uses the last completed validation checkpoint, repeating any steps after it. A crash before the first checkpoint restarts from the identical initial state; a crash during finalization reconstructs the final summary and seal from the completed checkpoint. Do not edit sources or change the recipe in a run directory.

Only after the three training runs finish, explicitly evaluate their sealed selections:

```text
conda run --no-capture-output --prefix ./.conda python experiment.py evaluate --run-dir runs/pilot/sketch_seed0_erm
conda run --no-capture-output --prefix ./.conda python experiment.py evaluate --run-dir runs/pilot/sketch_seed0_rsc
conda run --no-capture-output --prefix ./.conda python experiment.py evaluate --run-dir runs/pilot/sketch_seed0_cue
```

The pilot is exploratory if its target results influence later decisions. A later matrix including the same target is not an untouched confirmatory test. Freeze the full plan before viewing target outcomes when confirmatory interpretation matters.

## Run the complete matched comparison

To run the prespecified matrix without an exploratory target pilot:

```text
conda run --no-capture-output --prefix ./.conda python experiment.py matrix --seeds 0 1 2 --evaluate
```

All training settings in the JSON can also be overridden with explicit kebab-case arguments. For example, use `--learning-rate 6e-5 --batch-per-domain 12 --eval-batch-size 48 --precision bf16 --steps 5000 --eval-every 200 --workers 4` on `train`, `matrix`, and `profile`. Boolean fields support both forms, such as `--freeze-bn` / `--no-freeze-bn`. JSON settings are loaded first, then explicit flags are applied, and the resulting complete recipe is validated and saved in run contracts. Use matching settings for profiling and training; a changed recipe or source requires a new run directory.

This executes **36 runs sequentially**: four targets × three seeds × three methods. It completes every training run before accessing any target evaluation. Repeating the command resumes unfinished training and verifies/reuses completed selections and target artifacts without calling training again. Input seed order is canonicalized. The plan binds the dataset ID, and every arm’s initial weights are checked before target access. Omit `--evaluate` to finish all training while withholding target results.

```text
conda run --no-capture-output --prefix ./.conda python experiment.py report --runs-dir runs/pacs_resnet18
```

The report rejects incomplete or unmatched matrices. It saves target accuracies by domain, the four-domain mean per seed, mean and SD across seeds, paired cue-minus-RSC differences, NLL, ECE, and a ranking. Accuracy/SD tables use percentage points. Three seeds provide an initial comparison, not a general superiority claim.

## Files and evidence

| File | Purpose |
|---|---|
| `src/cue_routing/routing.py` | Exact detached cue-wise gradient routing |
| `src/cue_routing/models.py` | ResNet-18 and reusable CNN/ViT readout |
| `src/cue_routing/rsc.py` | Pinned DomainBed-style RSC operator |
| `src/cue_routing/data.py` | Official splits, manifests, balanced deterministic batches |
| `src/cue_routing/engine.py` | Matched training, exact resume, sealed evaluation |
| `configs/pacs_resnet18.json` | Complete primary experiment configuration |
| `METHOD.md` | Equations, ViT interface, assumptions, and failure modes |
| `PROTOCOL.md` | Data roles, controls, selection, and interpretation |
| `SOURCES.md` | Word-for-word evidence and pinned repositories |
| `verification/` | Actual machine-generated test and smoke reports |

After generating the verification reports, `conda run --prefix ./.conda python audit.py` checks syntax, the primary recipe, pinned split blobs, and whether the evidence matches the current executable sources. `conda run --prefix ./.conda python bundle.py` creates a portable full-project ZIP beside this project, including the classification and Office-Home workflows and shared requirements files. The ZIP includes code, configs, references, docs, and verification reports; it excludes the Conda environment, pretrained cache, PACS images, and run checkpoints.

Run artifacts include `contract.json`, `environment.json`, `history.json`, `last.pt`, `best.pt`, `selection.json`, `summary.json`, `target_access.json`, `target_metrics.json`, and `target_predictions.pt`. `best.pt` is finalized only after the planned training steps. Target predictions retain image paths for paired analysis. The files provide reproducibility and accidental-change detection, not a tamper-proof security boundary against someone rewriting the whole record.

The full review, repaired defects, regression evidence, and remaining limits are documented in [REVIEW.md](REVIEW.md). To repeat the static checks, install `requirements-dev.txt` inside the same Conda environment and run `ruff check src tests bootstrap.py experiment.py audit.py bundle.py` plus `ruff format --check src tests bootstrap.py experiment.py audit.py bundle.py`.

No full ViT benchmark is included in this first ResNet-18 experiment. The shared readout already accepts explicit patch-token grids, and numerical tests verify its CNN/token equivalence. A ViT experiment must specify its pooling rule and retrain every control consistently; see [METHOD.md](METHOD.md).
