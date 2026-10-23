# Controlled classification experiments

This directory implements CIFAR-100, CIFAR-100-C, Tiny ImageNet-200 with explicit 64px or 32px inputs, and the declared MNIST-CIFAR cue-conflict experiment. It shares the project environment and core operators with PACS and Office-Home. Each study records its own resolved recipe and source identity.

The main comparison is ERM, uniform non-target label smoothing, fixed global-to-local self-distillation, Zipf's Label Smoothing dense2, and CUE. These are training interventions on a common small-image ResNet-18. Final inference is global average pooling and the shared final classifier. CUE does not adapt or choose cues during testing.

## Select PostAct or PreAct

Every classification training mode accepts `--activation postact` or `--activation preact`: `erm`, `cue`, `ablation`, and full `train`/`study`. The same choice is supported by plans, CUDA profiles, and smoke verification. Omission preserves `postact`; all shipped presets explicitly retain that default. Set `"activation": "preact"` inside a JSON preset's `training` object to change its default. Dataset overrides may select a different backbone per dataset; an explicit CLI flag overrides both the common setting and all dataset overrides.

- `postact`: the existing torchvision ResNet-18 with a 3x3 stride-one stem and no maxpool. The selector preserves its original weights, state keys, RNG sequence, and predictions for the same seed.
- `preact`: the backbone from the pinned [Zipf author code, revision 108e27a](https://github.com/megvii-research/zipfls/blob/108e27aaf81aee0dc6dcd965ed6389f2ea06ba10/pytorch_zipfls/cifar_models/resnet.py). It uses a Conv/BN/ReLU stem, eight pre-activation blocks, and no terminal BN/ReLU. In this specific reference, even the no-projection shortcut uses the **preactivated** tensor. It is not a generic identity-shortcut ResNet-v2. Backbone and final-head initialization follow that reference; the optional Zipf auxiliary head keeps our isolated initialization to preserve matched ERM/CUE/Zipf shared weights.

Both expose the same 256/512-channel penultimate/final maps and support CIFAR 32px, Tiny 32/64px, and rectangular Dominoes inputs. The readout uses global spatial averaging, generalizing the reference's fixed 32px pooling to those sizes. Negative PreAct feature values are valid for CUE. There is no inference-time routing or new CUE parameter. This option compares two backbone implementations, including their initialization and shortcuts; it does not isolate only the position of BN/ReLU.

Use a separate study directory for each activation. Plans, run contracts, checkpoints, CUDA preflight signatures, startup logs, runtime snapshots, and comparison reports record the selected architecture. Changing activation cannot reuse an old preflight, resume another architecture's checkpoint, or silently overwrite an existing study. Existing checkpoints still require their original executable source; a source update is not a checkpoint migration.

For a fixed-recipe paired comparison, run the following after updating through GitHub and activating the existing project Conda environment. Change only `ACTIVATION` to choose the other backbone. This example schedules **four runs**: CIFAR-100 ERM/CUE with seeds 0 and 1, with no tuning or corruption grid. The first CUDA preflight measures the selected backbone's actual throughput and memory before training; no PreAct speed or accuracy improvement is assumed.

```bash
(
set -e
set -o pipefail
cd /workspace/cog_vision_model
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/.conda"
python bootstrap.py --device cuda --verify-only

ACTIVATION=preact
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="erm_cue_${ACTIVATION}_cifar100_bf16_b1024_seeds0_1_${STAMP}"
mkdir -p results

python -u classification/run.py ablation \
  --config classification/configs/ablation_5090.json \
  --data-root data/classification \
  --runs-dir "runs/${RUN_NAME}" \
  --activation "$ACTIVATION" \
  --datasets cifar100 \
  --methods erm cue \
  --seeds 0 1 \
  --no-tune \
  --epochs 200 \
  --milestones 100 150 \
  --batch-size 1024 \
  --eval-batch-size 2048 \
  --learning-rate 0.8 \
  --warmup-epochs 5 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --lr-decay 0.1 \
  --gate-tau 0.15 \
  --gate-temperature 0.05 \
  --joint-floor 0.5 \
  --precision bf16 \
  --eval-precision fp32 \
  --device cuda \
  --workers 8 \
  --max-vram-gib 28 \
  --logging-steps 200 \
  2>&1 | tee "results/${RUN_NAME}.log"
)
```

Use `--datasets tiny_imagenet --tiny-imagenet-size 32` for the paired Tiny32 comparison, or `64` for native-resolution Tiny. `--datasets cifar100 tiny_imagenet` schedules eight runs with two methods and two seeds. `--dry-run` prints the exact scope without starting training. For tuning, retain `--activation` but use `--tune --lr-candidates ... --tau-candidates ...` and remove scalar `--learning-rate`/`--gate-tau` overrides; selection remains validation-only. Keep each architecture's ERM and CUE controls matched.

## Trainer-style progress logging

The classification runners now use a run-wide optimizer-step progress bar, following the presentation used by the decoder VA project's Transformers Trainer. It is enabled by default for `erm`, `cue`, `ablation`, and full `train`/`study` runs, including output piped through `tee`. Training, validation, final-test, and CUDA preflight displays identify the dataset, method, and seed where applicable.

```text
Train tiny_imagenet/cue/s0:  25%|##5       | 4400/17600 [..., epoch=50.00/200, loss=..., lr=..., stage=train]
Eval tiny_imagenet/cue/s0 epoch=50: ...
[train] step=4600/17600 epoch=52.273 loss=... accuracy=... lr=...
[epoch] tiny_imagenet/cue/s0 epoch=53/200 step=4664/17600 train_loss=... train_accuracy=... eval_loss=... eval_accuracy=... lr=... images/s=... peak_allocated=... peak_reserved=... elapsed=... run_remaining~=... train_ce=... local_consistency=... joint_fraction=... gate_std=...
[study] stage=training runs=... train_steps=... clean=... remaining~=... finish_KST~=...
```

These are format examples, not measured performance. An optimizer step is counted only after a successful update; internal FP16 overflow retries do not advance it. `eval_loss` is global classification NLL, not CUE's combined classification-plus-consistency training objective. For 90,000 Tiny training images at batch 1,024, the loader uses **88** batches per epoch and **17,600** steps over 200 epochs. CIFAR uses **44** batches and **8,800** total steps. The code derives these counts from the loader budget, including the final partial batch and explicit debug limits.

Optional presentation flags:

```bash
--progress bar \
--logging-steps 200 \
--progress-interval 1
```

`bar` renders a live tqdm display and is the default. `plain` emits newline snapshots without carriage returns or cursor controls; use `--progress plain --progress-interval 10` for a clean text log. `off` hides progress displays while retaining periodic metrics, epoch summaries, study summaries, and saved telemetry. The refresh interval is in seconds; the logging interval counts accepted optimizer steps. These flags are presentation settings, separate from frozen training hyperparameters, and can change on a resume with the same executable source.

Each run writes `training_log.jsonl` with sample-weighted training windows, per-epoch training/validation metrics, learning rate, timing, memory, and session-end status. A fresh session ID and `resumed_from_step` identify each restart: records from an interrupted attempt are retained, while the checkpoint remains authoritative for completed training. The bar starts at the checkpoint's accepted step count, and its rate uses only newly executed work. Failures close the bar at the last accepted step. Complete runs reused from disk are not retrained.

The epoch line explicitly distinguishes online augmented `train_loss`/`train_accuracy` from deterministic internal-validation `eval_loss`/`eval_accuracy`. CUE also displays its global training CE, weighted local KL, mean joint gate, and mean within-image area-weighted gate standard deviation. Histories additionally record the area-weighted fraction of gates at least 0.99 (`joint_saturated_fraction`); this is a diagnostic threshold, not a training parameter. These statistics help distinguish an active spatial gate from constant or near-ERM behavior, but do not establish a performance benefit.

Each new CUDA run releases inactive cache from preflight/earlier runs before constructing its model and resetting the memory measurement window. `peak_allocated` measures live tensor allocation peaks, while `peak_reserved` includes the allocator's cached memory and remains the enforced budget. Both are cumulative peaks for the current run, including validation. The older repeated 23.31 GiB reservation could include a previous Tiny ImageNet preflight's cache and must not be interpreted as CIFAR's tensor-memory requirement.

The bar's remaining time estimates pending optimizer steps from this session's observed rate. The epoch's `run_remaining~` includes observed validation/checkpoint overhead. The separate `[study]` ETA includes all declared runs and final evaluations, with provisional dataset rates identified in `progress.json`. Finishing the training bar does not mean the final tests have completed. Detailed host-wait, validation, and checkpoint timings remain in the JSON telemetry rather than a second permanent console line every epoch.

`tqdm==4.70.1` is now an explicit dependency in `requirements.txt` (and is included through the CUDA/CPU requirements files). Install requirements in the existing project Conda environment if needed. Update source only between experiments: an existing process does not acquire new logging code, and the source-bound resume guard still rejects checkpoints from different executable versions.

## ERM first, then Tiny ImageNet at 32px

`classification/run.py erm` trains only the plain cross-entropy ResNet-18 baseline. The [`erm_5090.json`](configs/erm_5090.json) preset uses the same fixed optimizer, architecture, initialization, data split, augmentation, batch sizes, precision, and schedule as the original CUE-only 5090 preset. Its four runs are CIFAR-100 32px and Tiny ImageNet 64px, each with seeds 0 and 1. It does not rerun the completed CUE jobs. Compare its `erm_measured` results with the original `cue_measured` results, retaining both reports and their source identities; this command does not import, alter, or claim to have revalidated the historical checkpoints.

The separate [`tiny32_ablation_5090.json`](configs/tiny32_ablation_5090.json) preset runs ERM and CUE at 32px on Tiny ImageNet, using the same two seeds and fixed hyperparameters. This is four additional runs, with paired CUE-minus-ERM differences in the report. Both presets disable tuning: the combined workload is **8 runs / 1,600 epochs**, with no hidden development, mechanism, or corruption jobs. Keep batch 1,024 and LR 0.8 for this comparison; raising the batch merely to fill VRAM would add another changed variable.

`--tiny-imagenet-size 32` applies Pillow bilinear resizing before augmentation and normalization in **training, internal validation, and final evaluation**. The 32px crop has zero padding 4 and the 64px crop has padding 8, preserving the relative augmentation width. CIFAR processing is unchanged. The original 64px dataset files, split identities, and labels remain unchanged, so prepared data need not be downloaded or converted again. Resolution is stored in every recipe/contract, bound into preflight signatures, printed in the run log, and reported using the actual model input size. A different resolution requires a new run directory.

Finish any currently running training before updating executable sources. Update through GitHub, install the pinned requirements in the existing environment, and verify CUDA:

```bash
(
set -e
set -o pipefail
cd /workspace/cog_vision_model
git pull --ff-only origin main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/.conda"
python -m pip install -r requirements-cuda.txt
python bootstrap.py --device cuda --verify-only
)
```

Run the native-resolution ERM baselines first:

```bash
(
set -e
set -o pipefail
cd /workspace/cog_vision_model
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/.conda"
mkdir -p results
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="erm_native_bf16_b1024_seeds0_1_${STAMP}"

python -u classification/run.py erm \
  --config classification/configs/erm_5090.json \
  --data-root data/classification \
  --runs-dir "runs/${RUN_NAME}" \
  --datasets cifar100 tiny_imagenet \
  --seeds 0 1 \
  --tiny-imagenet-size 64 \
  --epochs 200 \
  --milestones 100 150 \
  --batch-size 1024 \
  --eval-batch-size 2048 \
  --learning-rate 0.8 \
  --warmup-epochs 5 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --lr-decay 0.1 \
  --precision bf16 \
  --eval-precision fp32 \
  --device cuda \
  --workers 8 \
  --max-vram-gib 28 \
  2>&1 | tee "results/${RUN_NAME}.log"
)
```

After that finishes, run both arms at 32px:

```bash
(
set -e
set -o pipefail
cd /workspace/cog_vision_model
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/.conda"
mkdir -p results
STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="tiny32_erm_cue_bf16_b1024_seeds0_1_${STAMP}"

python -u classification/run.py ablation \
  --config classification/configs/tiny32_ablation_5090.json \
  --data-root data/classification \
  --runs-dir "runs/${RUN_NAME}" \
  --datasets tiny_imagenet \
  --methods erm cue \
  --seeds 0 1 \
  --no-tune \
  --tiny-imagenet-size 32 \
  --epochs 200 \
  --milestones 100 150 \
  --batch-size 1024 \
  --eval-batch-size 2048 \
  --learning-rate 0.8 \
  --warmup-epochs 5 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --lr-decay 0.1 \
  --gate-tau 0.15 \
  --gate-temperature 0.05 \
  --joint-floor 0.5 \
  --precision bf16 \
  --eval-precision fp32 \
  --device cuda \
  --workers 8 \
  --max-vram-gib 28 \
  2>&1 | tee "results/${RUN_NAME}.log"
)
```

Append `--dry-run` to either Python command to inspect its inputs and exact budget. The GPU preflight measures only the requested methods at their real input sizes, and observed epochs update the remaining-time estimate. Actual throughput and memory on the 5090 must be measured there; 32px is not promised to be four times faster. Outputs are `runs/<RUN_NAME>/comparison.json`, `comparison.md`, `progress.json`, and `preflight/`. For a resume, reuse the original run name, identical settings, and unchanged executable sources, and append logs with `tee -a`.

## Matched ResNet ablation and bounded tuning

Use `classification/run.py ablation` to compare CUE with the same small-image ResNet-18 trained using only cross-entropy (`erm`). The editable inputs are in [`configs/ablation_5090.json`](configs/ablation_5090.json). The default starts with **CIFAR-100 only**: six development jobs plus four final jobs, totaling **10 runs / 2,000 epochs**. Adding Tiny ImageNet explicitly doubles this budget. Preflight and observed epoch timing estimate the complete declared workload automatically.

The sequence is fixed before training:

1. Train ERM at LR `0.1, 0.4, 0.8`, using development seed `2027` and the full 200-epoch schedule. Select by final-epoch internal-validation accuracy, then lower NLL, then candidate order.
2. Freeze that LR for every arm. Train CUE with thresholds `0.05, 0.15, 0.30` on the same development seed, selecting by the same validation rule. Temperature `0.05` and joint floor `0.5` remain fixed. This is a bounded sequential search, not an exhaustive optimization of all CUE parameters.
3. Freeze both selections and retrain ERM and CUE with seeds `0, 1`. Every paired arm uses the same initial weights, data split, augmentation recipe, batch, LR, warmup, epoch schedule, weight decay, and inference architecture. Only the loss intervention changes. The development seed must be separate from these final seeds.
4. Finish and seal all declared training before opening final tests. The report verifies prediction identities and includes per-seed accuracy, mean, seed SD, and paired CUE-minus-ERM differences in percentage points. Two seeds provide pilot evidence; the code makes no formal significance claim.

The ERM-first LR choice avoids weakening the baseline by choosing a CUE-specific common recipe. The tradeoff is that it does not find each method's independently optimal LR. A later larger search must be separately declared. Published reference scores remain descriptive comparisons: the backbone implementation, training split, batch, and Tiny ImageNet resolution differ.

Update through GitHub after stopping or finishing any active job in this checkout. Preserve existing run folders; source-bound jobs cannot resume after a source change. Then activate the existing Conda environment and launch:

```bash
(
set -e
set -o pipefail

cd /workspace/cog_vision_model
git pull --ff-only origin main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/.conda"
python bootstrap.py --device cuda --verify-only
mkdir -p results

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="cifar100_erm_cue_tuned_b1024_seeds0_1_${STAMP}"

python -u classification/run.py ablation \
  --config classification/configs/ablation_5090.json \
  --data-root data/classification \
  --runs-dir "runs/${RUN_NAME}" \
  --datasets cifar100 \
  --methods erm cue \
  --seeds 0 1 \
  --development-seed 2027 \
  --tune \
  --lr-candidates 0.1 0.4 0.8 \
  --tau-candidates 0.05 0.15 0.30 \
  --epochs 200 \
  --milestones 100 150 \
  --batch-size 1024 \
  --eval-batch-size 2048 \
  --warmup-epochs 5 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --lr-decay 0.1 \
  --gate-temperature 0.05 \
  --joint-floor 0.5 \
  --precision bf16 \
  --eval-precision fp32 \
  --device cuda \
  --workers 8 \
  --max-vram-gib 28 \
  2>&1 | tee "results/${RUN_NAME}.log"
)
```

Use `--dry-run` to inspect the budget without training. Use `--no-tune --learning-rate 0.8 --gate-tau 0.15` for a fixed-recipe four-run ablation, removing the scalar/grid ambiguity: scalar LR/tau flags are rejected while tuning is enabled. Five final seeds (`--seeds 0 1 2 3 4`) make 16 runs on one dataset with the default six tuning runs. Identical commands with the original `--runs-dir` resume/reuse the frozen study; changing scope or source requires a new directory.

To test adaptive gating beyond the overall regularization effect, use `--methods erm fixed_global_local cue`. It adds three validation weight trials (`--fixed-candidates 0.1 0.25 0.5`) and two final runs, yielding 15 total CIFAR jobs. CUE must improve on this control to support an adaptive-weighting benefit. It is not enabled silently. The original four-job CUE-only workflow remains available below.

## CUE first on RTX 5090

This fixed-recipe mode trains only CUE. The two natural datasets with seeds 0 and 1 require **4 runs / 800 epochs**, instead of the full study's 50 runs / 8,200 epochs. There is no development grid, competitor training, dominoes task, or corruption evaluation in this mode. CUE coefficients are fixed before training.

The editable preset is [`configs/cue_5090.json`](configs/cue_5090.json). It uses training batch 1,024, evaluation batch 2,048, BF16 training, FP32 evaluation, eight loader workers, and a **28 GiB peak-reserved memory ceiling**. This ceiling does not allocate or guarantee that amount of VRAM. Real CUE training/validation preflight checks each selected dataset at the declared batch sizes, and saves throughput and memory under the run's `preflight/` directory. An OOM or numerical failure stops before benchmark training without changing the batch.

The larger batch changes optimization. This preset explicitly sets peak SGD LR 0.8 and five epochs of linear warmup: 0.16, 0.32, 0.48, 0.64, 0.8. It retains 200 epochs and decays after epochs 100 and 150. This is an unverified large-batch recipe, not an established accuracy-preserving speedup. Batch size, LR, and warmup are independent inputs; changing one does not silently change the others. Training retains deterministic kernels and exact epoch-boundary resume.

Use the existing Conda environment and installed CUDA requirements; this mode adds no dependencies. For a new server, follow [the root setup instructions](../README.md#setup). If the old study is still running, stop it in its terminal before updating the source. Keep its run directory and start this different recipe in a new directory.

If the natural-image manifests are already prepared, proceed directly to training. Otherwise, prepare only these two datasets:

```bash
python -u classification/experiment.py prepare --dataset cifar100 --download \
  --data-root data/classification
python -u classification/experiment.py prepare --dataset tiny_imagenet --download \
  --data-root data/classification
```

Launch from Bash after putting the updated project files on the server:

```bash
(
set -e
set -o pipefail

cd /workspace/cog_vision_model
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/.conda"
mkdir -p results

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="cue_only_5090_bf16_b1024_seeds0_1_${STAMP}"

python -u classification/run.py cue \
  --config classification/configs/cue_5090.json \
  --data-root data/classification \
  --runs-dir "runs/${RUN_NAME}" \
  --datasets cifar100 tiny_imagenet \
  --seeds 0 1 \
  --epochs 200 \
  --milestones 100 150 \
  --batch-size 1024 \
  --eval-batch-size 2048 \
  --learning-rate 0.8 \
  --warmup-epochs 5 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --lr-decay 0.1 \
  --gate-tau 0.15 \
  --gate-temperature 0.05 \
  --joint-floor 0.5 \
  --precision bf16 \
  --eval-precision fp32 \
  --device cuda \
  --workers 8 \
  --max-vram-gib 28 \
  2>&1 | tee "results/${RUN_NAME}.log"
)
```

Add `--dry-run` to preview the resolved settings and run count without data/CUDA access. Explicit CLI values override both common JSON settings and dataset overrides. To run just one job, use `--datasets cifar100 --seeds 0`. To resume, repeat the same command with the **original fixed `--runs-dir`**, identical settings, and unchanged source/environment; do not generate a new timestamp. Completed jobs and predictions are verified and reused.

Every training job prints a `[run]` line containing its PID, actual batch sizes, precision, workers, and run directory. Each `[epoch]` summary prints accuracy, loss, LR, images/second, peak reserved VRAM, and elapsed/remaining time. `training_log.jsonl` records time waiting for data (including worker startup/teardown), step-and-host time, validation time, and checkpoint time. Step-and-host time includes transfers and Python work; it is not a CUDA-kernel-only measurement. `runtime_status.json` inside each individual run records the latest phase, PID, configuration, and current/peak allocator counters. A single `nvidia-smi` screenshot does not identify the recipe or establish sustained utilization. The memory ceiling does not force memory allocation.

`progress.json` and console output include elapsed time, remaining time, expected finish time, run/epoch counts, and per-dataset timing coverage. Preflight supplies provisional rates; completed epochs replace them with observed rates. No fixed 5090 completion time is claimed before measurement. Final evaluation starts only after all requested training jobs are complete.

Results are written to `comparison.json` and `comparison.md`. They include CUE accuracy by seed, mean and seed SD, and nine cited published reference methods. The reference data and exact paper excerpts are in [`references/published_classification.json`](references/published_classification.json). Those rows are labeled `published_reference`; only CUE rows are local measurements. The report records differences in backbone, input resolution, internal validation split, batch/LR, and seed count. Differences in percentage points are descriptive across protocols; there is no paired significance test or automatic superiority claim. Debug results are marked synthetic and omit accuracy-difference comparisons.

The entrypoints call `clvision/cue_settings.py` for inputs, `clvision/cue_workflow.py` for orchestration/reporting, and the existing `clvision/engine.py` for training. The full controlled study remains available below when it is needed.

## Full controlled study (optional)

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

`--seeds` sets the main and mechanism seed lists together. `--main-seeds` and `--mechanism-seeds` override their respective lists if supplied; duplicate, empty, or out-of-range lists are rejected. The default remains seeds 0 and 1, with development seed 2027: 18 validation-candidate runs, 20 natural-image main runs, and 12 mechanism runs, totaling 50 training runs and 8,200 epochs at the default schedules. The validation candidates each train for the full 200 epochs. Three seeds (`--seeds 0 1 2`) produce 66 runs with the default candidate grids. The preview calculates counts from your actual settings.

Precedence is built-in defaults, JSON common training settings, JSON dataset-family schedules, JSON dataset overrides, then explicit CLI flags. Natural-image `--epochs`/`--milestones` and mechanism `--mechanism-epochs`/`--mechanism-milestones` are separate. Supply compatible milestones when changing epochs; `--milestones` with no values disables the natural-image milestone schedule. Validation grids and dataset-specific coefficients can be edited in JSON; each dataset keeps an equal positive candidate budget across the three tuned methods.

The same resolved recipes drive CUDA preflight and training. Preflight checks every intervention using actual data, the requested training/evaluation batch sizes and precision, and ten accepted optimizer updates. A failed preflight stops the study. The workflow then develops on internal validation, freezes choices, trains all main/mechanism runs, seals the complete study, and evaluates the final checkpoints, including CIFAR-100-C. Results are written to the selected run directory as `comparison.json` and `comparison.md`.

Training precision (`--precision`, default `fp16`) and evaluation precision (`--eval-precision`, default `fp32`) are independent, recorded settings. FP32 evaluation applies consistently to preflight validation, every epoch's validation, clean tests, mechanism diagnostics, and corruption tests. Early training with BatchNorm running statistics can produce validation activations outside FP16's range even when training updates are finite. Evaluation therefore defaults to FP32, without an automatic per-batch precision fallback. Preflight measures its actual throughput and memory cost. Explicit `fp16` and `bf16` evaluation remain available; a precision change requires a new study directory.

Preflight failures retain `status=failed`, the dataset, method, stage, accepted update count, resolved recipe, and exception in `classification/verification/<dataset>_cuda.json`. Evaluation reports nonfinite logits separately from invalid label IDs and includes input/model-state diagnostics. Corrupted floating-point model buffers stop training before an optimizer update is accepted. Partially consumed preflight data loaders are shut down on failure so their worker/pin-memory errors do not obscure the original error. These checks do not clamp or replace invalid predictions.

Timing is automatic. Preflight prints warmed training/evaluation throughput for each dataset and method. All ten training updates are checked, but the first five are excluded from throughput timing. Evaluation warms up on the first full batch, then times up to ten batches, reusing that first batch once and streaming the rest. Loader construction and first-fetch time are recorded separately and charged once per estimated pass, rather than multiplied across every batch. Before the first full training run, `[study]` prints a provisional whole-study ETA based on those measurements. It includes development, all main/mechanism epochs, per-epoch validation, clean tests, mechanism panel diagnostics, and all 750 CIFAR-100-C checkpoint/condition evaluations under the default two-seed scope. Actual epoch times, including checkpoint writes, replace provisional rates as training proceeds; measured final evaluation times refine that stage's estimate. Final sealing, report generation, and downtime are not predicted. An unavailable measurement is shown as `estimating`, never an invented duration.

The `epoch_rates=observed/pending` counter reports how many remaining dataset/method groups have complete-epoch timing. For example, `1/16` means only one group's training rate has been calibrated with actual epochs; the other fifteen still use preflight. `progress.json` includes `estimated_remaining_by_dataset` and `provisional_training_timing_groups` so an unobserved dataset cannot silently dominate an apparently settled total. Full-epoch calibration for CIFAR does not calibrate Tiny ImageNet or final corruption evaluation. A short active-run ETA and a much larger provisional study ETA describe different scopes.

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

The original full-study preset uses torchvision's post-activation ResNet-18, a3x3 stride-one stem, no maxpool, trainable BatchNorm statistics, no pretraining, batch128, and SGD with learning rate0.1, momentum0.9, weight decay0.0001. The rate drops tenfold after epochs100 and150. All use the final epoch200 checkpoint. CIFAR uses padded32px crops and flips; Tiny uses padded64px crops and flips. The selectable PreAct option and 5090 presets are described above; the resolved run recipe is authoritative. The only extra Zipf parameters are its training-only penultimate linear classifier.

The Zipf port retains final-plus-penultimate votes, factor-two bilinear spatial upsampling, minimum-tie ranks, exponent1, target exclusion and renormalization, unmasked non-target KL, and auxiliary CE. Auxiliary weights are0.1 for CIFAR and0.5 for Tiny. Its chosen KL weight comes only from the declared validation candidates. The official reference's small-image backbone is pre-activation and its published Tiny experiments resize to32px; the original common post-activation/native64px study is a matched reevaluation, not numerical reproduction of the paper. Selecting PreAct aligns the backbone block/stem implementation but does not match the complete data, optimizer, auxiliary initialization, and evaluation protocol. The full pinned reference files, license, and hashes are under `references/`.

CUE uses the current per-cue detached disagreement gate and its exactly gradient-equivalent global-to-local KL expression. The numerical implementation checks gradients through both shared features and classifier parameters. Its scalar logged loss need not equal the legacy surrogate's scalar value. A gain over fixed consistency is required to attribute usefulness to adaptive weighting.

The default development grid has three final-epoch validation candidates per tunable method, using training seed2027. CUE thresholds are0.05/0.15/0.30 at temperature0.05 and joint floor0.5; fixed KL weights are0.1/0.25/0.5; Zipf KL weights are0.03/0.1/0.3 on CIFAR and0.3/1/3 on Tiny. Validation selection prefers accuracy, then lower NLL, then registered candidate order. Candidate completion hashes are bound into the frozen selection and main-run contracts.

## Evidence and limitations

The primary question is whether the adaptive method improves on fixed consistency on both natural datasets. ERM and label smoothing test practical value; Zipf is the selected external comparator. All seeds and methods are reported. With only two pilot seeds, bootstrap intervals are descriptive and unstable; the code does not claim statistical significance or2026 state of the art. A corruption-only gain with lower clean accuracy is a tradeoff. A mechanism-only gain does not establish general image classification utility.

The completed PACS result remains a negative ERM comparison. Existing Office-Home training remains supplementary DG evidence. This code does not implement a new ViT benchmark or prove the cognitive interpretation of final feature-map regions.

See `REVIEW.md` for implementation checks, `SOURCES.md` for exact paper excerpts, and generated `verification/` reports for the source-bound outcomes actually measured on the executing machine.

The default settings retain the two-seed pilot. Existing study plans and checkpoints cannot be resumed across changed source or scope. Finish or stop an existing training process before updating code; keep its original results as a separate study.

The audit retains historical PACS/Office-Home hash comparisons as provenance information; current verification still requires matching current source hashes and successful outcomes. Update the full project through GitHub between experiments, using `git pull --ff-only origin main` from the server checkout. Local edits must be preserved and reconciled before pulling; no ZIP extraction is needed.
