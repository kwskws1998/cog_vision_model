# Command-line experiments

Use `classification/run.py tune` and put experiment inputs in the command.
Changing hyperparameters, search values, seeds, ordering, or a deadline does not
require editing a Python or JSON file. The same command is available through
`classification/experiment.py tune`; both use the existing `Comparison` engine.

## Additional CUE temperature candidates

This command reuses the prepared CIFAR-100 data and existing Conda environment.
It trains two new CUE candidates with seed 0, without retraining ERM or the
fixed-gate control. The search varies only temperature (0.025 and 0.10); tau
is 0.30 and the joint floor is 0.5.

```bash
(
set -e
set -o pipefail

cd /workspace/cog_vision_model
git pull --ff-only origin main

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/.conda"
export CUDA_VISIBLE_DEVICES=0

python bootstrap.py --device cuda --verify-only

mkdir -p results
RUN_NAME="cifar100_preact_cue_temperature_fp32_b128_seed0_$(date +%Y%m%d_%H%M%S)"

python -u classification/run.py tune \
  --data-root data/classification \
  --runs-dir "runs/${RUN_NAME}" \
  --dataset cifar100 \
  --seeds 0 \
  --activation preact \
  --initialization msra \
  --normalization zipfls \
  --epochs 200 \
  --milestones 100 150 \
  --batch-size 128 \
  --eval-batch-size 128 \
  --learning-rate 0.1 \
  --warmup-epochs 0 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --lr-decay 0.1 \
  --gate-tau 0.30 \
  --gate-temperature 0.05 \
  --joint-floor 0.5 \
  --gradient-clip 0 \
  --precision fp32 \
  --eval-precision fp32 \
  --device cuda \
  --workers 4 \
  --max-vram-gib 28 \
  --tune-gate-tau 0.30 \
  --tune-gate-temperature 0.025 0.10 \
  --tune-joint-floor 0.5 \
  --search-strategy grid \
  --search-seed 2027 \
  --max-trials 2 \
  --selection-metric accuracy \
  --no-include-baseline \
  --reference classification/references/cifar100_seed0_cue_temperature005.json \
  --stages preflight tune evaluate \
  --preflight-steps 10 \
  --reserve-minutes 10 \
  --time-safety-factor 1.25 \
  --progress plain \
  --logging-steps 200 \
  --progress-interval 30 \
  2>&1 | tee "results/${RUN_NAME}.log"
)
```

The scalar gate values define the base recipe. Each `--tune-...` list overrides
that scalar for the candidates. Omit all three tuning lists to run one fixed CUE
candidate. No experiment-specific configuration file is loaded by this command.

## Change inputs in the command

| Argument | Behavior |
|---|---|
| `--seeds 0` or `--seeds 0 1` | Explicit training seeds; each has an independent search |
| `--dataset cifar100` / `tiny_imagenet` | Dataset; use `--normalization dataset` for Tiny ImageNet |
| `--tiny-imagenet-size 32` / `64` | Tiny ImageNet input resolution |
| `--tune-gate-tau`, `--tune-gate-temperature`, `--tune-joint-floor` | Lists forming the Cartesian candidate grid |
| `--search-strategy grid` | Declared value order, nested tau, temperature, floor |
| `--search-strategy random --search-seed 2027` | Reproducibly shuffle the finite grid; no extra training seeds |
| `--max-trials 2` | At most two candidates per seed after ordering; omit for the whole grid |
| `--selection-metric accuracy` / `nll` | Final-epoch validation selection criterion |
| `--include-baseline` | Add one matched ERM run per seed |
| `--no-include-baseline` | CUE-only search |
| `--stages preflight tune baseline evaluate` | CUE tuning first, then ERM |
| `--stages preflight baseline tune evaluate` | ERM first, then CUE tuning |
| `--stages preflight tune evaluate` | CUE-only execution |
| `--finish-by 2026-10-25T10:30:00+09:00` | Optional explicit deadline; omission means no deadline |
| `--reserve-minutes 10 --time-safety-factor 1.25` | Candidate admission margin when a deadline is set |
| `--cpu-threads N` | Optional PyTorch CPU thread count; omission preserves the runtime setting |
| `--progress bar` / `plain` / `off` | Progress presentation |
| `--dry-run` | Validate and print the full plan without creating runs or reading datasets |
| `--resume` | Reuse the original output directory and unchanged resolved experiment |

Use a fresh `--runs-dir` after changing inputs. For resume, keep the original
directory rather than generating a new timestamp. A changed recipe, candidate
list, budget, executable source, or dataset identity is rejected. Completed final
predictions are verified and reused instead of testing again.

A deadline controls admission of another complete candidate; an admitted trial
still finishes its full epoch schedule. Preflight and epoch logs provide measured
time estimates, not a guaranteed finish. With multiple seeds, admission reserves
time for the current comparison, not all future seeds. No deadline is enabled in
the command above.

## Existing validation reference and test separation

The [reference](references/cifar100_seed0_cue_temperature005.json) records the
previous seed-0 CUE configuration, validation accuracy 0.7494, dataset identity,
initial shared-weight hash, and verified archive provenance. It contains no test
score and loads no historical checkpoint; the server needs no old ZIP or run
directory. This file is historical evidence, not a source of new hyperparameters.

The reference requires one matching seed, common recipe, split identity, validation
count, and initialization. If you change the dataset, common training settings, or
seed, omit `--reference` or provide a matching reference. Also omit it for a new
paired ERM/CUE comparison. Do not edit its old measurements to force a match.

With `--selection-metric accuracy`, a new candidate must strictly exceed the
reference's **74.94% validation accuracy** to replace it. An accuracy tie keeps
the reference, even with lower NLL. Among new candidates, accuracy, NLL, then
registered order break ties. NLL selection requires strictly lower reference NLL.

CIFAR-100 has 45,000 training examples, 5,000 validation examples, and the separate
official 10,000-example test set. Training updates use train; epoch evaluation and
candidate selection use validation. Test arrays remain unopened until selection
is frozen. If the reference wins, no new test is run. Otherwise the selected new
candidate's final checkpoint is reused without retraining and evaluated once.
Unselected development candidates never access test. An explicitly requested ERM
baseline is evaluated only after training and selection close.

## Artifacts and verification

For one seed, the output directory contains `comparison.md`, `comparison.json`,
`plan.json`, `search.json`, `selection.json`, and `invocations.jsonl`. The last
file records the exact command arguments, stage order, and actual CPU thread count.
The plan freezes resolved candidate configurations and source identity.

`development/cueN/` holds each candidate's history and validation summary.
A selected new winner has `main/seed0_cue/test_metrics.json`; otherwise the report
explicitly states that no new test was performed. Multiple seeds use subdirectories
`seedN/`, each with its own comparison report.

For a small synthetic CPU execution, use `--smoke` and omit `--reference`.
The printed plan explicitly shows its two-epoch, small-batch CPU overrides.
This verifies execution and selection, not benchmark quality or CUDA throughput.
Production preflight uses the requested actual train/validation configuration.

The Python APIs `Comparison`, `Search`, `Budget`, and `SingleRun` remain available
for custom orchestration. Older scripts under `classification/experiments/` are
optional examples; the CLI above does not import their experiment constants.
