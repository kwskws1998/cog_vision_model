# Editable Python experiments

The executable experiment is
[experiments/erm_cue.py](experiments/erm_cue.py). Edit its input block to change
the recipe or experiment order. The training engine, CUE implementation, and
comparison orchestration remain reusable modules.

| Input | What it controls |
|---|---|
| `COMMON` | Activation, initialization, normalization, epochs, optimizer, batch sizes, precision, workers, and memory limit shared by ERM and CUE |
| `DATASETS`, `DATASET_OVERRIDES` | Datasets to run and explicit per-dataset changes to common settings |
| `SEEDS` | Explicit training seeds; each seed creates its own matched comparison |
| `TUNING` | Enable/disable tuning, grid/random order, trial limit, sampling seed, and validation selection metric |
| `CUE_GRID`, `FIXED_CUE` | Candidate parameter values, or one fixed CUE recipe when tuning is disabled |
| `STAGES` | Order of the preflight, tuning, baseline, and final evaluation stages |
| `BUDGET` | Optional timezone-aware deadline, reserved minutes, and timing safety factor |
| `PREFLIGHT_STEPS`, `DISPLAY` | Preflight updates, progress display, and logging frequency |

There is no mandatory 200-epoch, batch-128, FP32, PreAct recipe in the generic
comparison API. Those values are visible example inputs. Configuration validation
checks supported values and consistency, such as milestones falling within the
training schedule; it does not force the example's numeric settings.

## Change the experiment inputs

All candidates inherit the exact `COMMON` settings and current seed. The search
varies `gate_tau`, `gate_temperature`, and `joint_floor`; these are CUE parameters.
Changing the learning rate, schedule, precision, or backbone in `COMMON` applies
the same change to both ERM and CUE. `Search(candidates=[{}])` uses the CUE values
already declared in `COMMON` for a fixed-setting comparison without a parameter
sweep. Candidate dictionaries can override some or all three CUE fields; omitted
fields inherit the common input. More candidates add training runs, not seeds.

The checked-in example has one dataset, one training seed, and six CUE candidates:
`3 gate_tau values × 2 joint_floor values × 1 gate_temperature value`. Its maximum
workload is therefore six CUE training runs and one ERM run. `TUNING.max_trials`
limits the candidate count; `None` retains the whole declared grid. Setting
`TUNING.enabled=False` runs only `FIXED_CUE` plus ERM.

`strategy="grid"` uses the grid's declared order. `strategy="random"` shuffles the
same finite grid with `sampling_seed` before truncating to `max_trials`; it does
not introduce new model seeds or an unbounded search. Set `TUNING.metric` to
`"accuracy"` or `"nll"`. For Tiny ImageNet, include `"tiny_imagenet"` in `DATASETS`;
the example explicitly sets that dataset's normalization through
`DATASET_OVERRIDES`. Change `tiny_imagenet_size` to choose 32px or 64px input.

The candidate sequence determines admission priority when a deadline is active.
Define the sequence directly or generate it in ordinary Python:

```python
CANDIDATES = [
    {"gate_tau": tau, "gate_temperature": temperature, "joint_floor": floor}
    for floor in (0.5, 0.0)
    for temperature in (0.05,)
    for tau in (0.15, 0.05, 0.30)
]
```

`Search(metric="accuracy", candidates=CANDIDATES)` selects by highest validation
accuracy. `metric="nll"` selects by lowest validation negative log-likelihood.
The metric is declared before the comparison starts; neither option uses test scores.

For CUE tuning first:

```python
STAGES = ("preflight", "tune", "baseline", "evaluate")
```

For ERM first, only reorder the middle two stages:

```python
STAGES = ("preflight", "baseline", "tune", "evaluate")
```

Preflight precedes full training. Final evaluation requires both a completed ERM
baseline and a frozen CUE selection. Reordering stages cannot bypass that data
boundary. A deadline controls whether another full candidate may start; it does
not silently lower the epoch count. Set `finish_by=None` to disable the deadline.
The supplied experiment's `BUDGET.finish_by` is
`"2026-10-24T10:30:00+09:00"`, preserving the requested October 24, 10:30 KST target
as an editable recipe input. The reusable `Budget` class itself defaults to no
deadline. For another run date, enter a complete future ISO timestamp including its
timezone; the program does not silently advance expired dates.

## Reusable comparison API

The following Python code runs one comparison from the repository root with
already prepared data. All relevant common training inputs are explicit. The
checked-in experiment wraps this API with data preparation, multiple dataset/seed
jobs, snapshots, and the convenience command-line options below.

```python
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "classification")]

from clvision.config import Config
from clvision.experiment import Comparison
from clvision.experiment_settings import Budget, Search, cue_grid

COMMON = dict(
    dataset="cifar100", method="erm", seed=0,
    activation="preact", initialization="msra", normalization="zipfls",
    tiny_imagenet_size=64,
    epochs=200, batch_size=128, eval_batch_size=128,
    learning_rate=0.1, warmup_epochs=0,
    momentum=0.9, weight_decay=1e-4,
    milestones=(100, 150), lr_decay=0.1,
    gate_tau=0.15, gate_temperature=0.05, joint_floor=0.5,
    gradient_clip=0.0, precision="fp32", eval_precision="fp32",
    device="cuda", workers=4, max_vram_gib=28.0,
)
SEARCH = Search(
    candidates=cue_grid(
        joint_floor=[0.5, 0.0],
        gate_tau=[0.15, 0.05, 0.30],
        gate_temperature=[0.05],
    ),
    metric="accuracy",
)
STAGES = ("preflight", "tune", "baseline", "evaluate")
BUDGET = Budget(finish_by=None, reserve_minutes=20, time_safety_factor=1.25)
RUN_DIRECTORY = ROOT / "runs" / datetime.now().strftime("erm_cue_%Y%m%d_%H%M%S_%f")

with Comparison(
    root=ROOT,
    data_root=ROOT / "data/classification",
    directory=RUN_DIRECTORY,
    config=Config(**COMMON),
    search=SEARCH,
    budget=BUDGET,
    resume=False,
    preflight_steps=10,
) as run:
    run.run(stages=STAGES)
```

The same context supports direct sequential calls:

```python
run.preflight()
run.tune()
run.baseline()
run.evaluate()
```

`run.run(...)` and the individual methods apply the same dependency checks. The
execution record preserves the requested order and stage journal.

The checked-in experiment handles paths and command-line options. Copy it to a new
experiment file when saving another recipe. The resolved inputs are persisted in
the run's plan, so a short command still records the exact scientific settings.
The checked-in runner saves its recipe as `RUNS_DIR/experiment.py`. Editable
experiment scripts are excluded from the training engine's source identity, so
editing an unrelated recipe does not invalidate an existing run. Resuming still
requires the same resolved inputs and unchanged executable training modules.

## Install and execute

Use the existing project Conda environment. The refactor adds no package dependency:

```bash
cd /workspace/cog_vision_model
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PWD/.conda"
python bootstrap.py --device cuda --verify-only
```

Edit the Python input block, then inspect the full resolved plan without training:

```bash
python classification/experiments/erm_cue.py --dry-run
```

Exercise the orchestration on a small synthetic CPU fixture:

```bash
python classification/experiments/erm_cue.py \
  --smoke \
  --runs-dir runs/erm_cue_synthetic_check
```

Synthetic checks establish control flow, not CUDA speed or benchmark accuracy. The
smoke uses the selected datasets, seeds, and candidate list; with the example's
defaults it executes all six CUE candidates and ERM on a reduced two-epoch fixture.
It does not replace or reduce the registered candidate grid. Smoke explicitly
disables the deadline so it remains usable after the recipe's target date; it does
not simulate production deadline admission. The
production `preflight` stage exercises every registered CUE configuration and ERM
on real train/validation batches before full training. Its update count is an
explicit experiment input. A failed preflight stops the comparison. Short checks
cannot guarantee that a later epoch will never fail.

Run the actual experiment with a fresh output directory:

```bash
python -u classification/experiments/erm_cue.py \
  --runs-dir runs/cifar100_erm_cue_trial01
```

Omitting `--runs-dir` generates a fresh timestamped directory. Use `--data-root`
to point to prepared data elsewhere, and `--download` when the requested dataset
has not been prepared. Before a production run, check the recipe's October 24,
10:30 KST deadline and update it for the intended date, or set `finish_by=None`
to run without a time limit.

The Python file is the experiment code. Additional experiments can call the same
API with their own ordinary Python values; changing a trial does not require
editing library files or adding another dedicated runner.

## Data and result interpretation

CIFAR-100 uses the prepared deterministic stratified split: 45,000 train examples,
5,000 validation examples, and the separate official 10,000-example test set.
Training updates use train only. Epoch logging and candidate selection use
validation only. Final test evaluation happens after training and search closure,
for ERM and the selected CUE model. The chosen CUE checkpoint is reused without
another training run. The plan records the actual audited manifest counts.

Common training settings and paired initial model weights are checked before final
evaluation. Each comparison writes its plan, training histories, timing admission
journal, frozen selection, final-test predictions, and comparison report. Candidate
folders remain validation-only artifacts. The report states actual final-test
metrics separately from validation selection metrics.

The checked-in example saves each dataset/seed comparison under
`RUNS_DIR/DATASET/seedN/`. For the default job, the final reports are
`RUNS_DIR/cifar100/seed0/comparison.md` and `comparison.json`. `plan.json`,
`execution.json`, `search.json`, and `selection.json` live in the same comparison
directory; `RUNS_DIR/experiment.py` holds the input script snapshot.

When tuning precedes ERM, admitting another CUE candidate reserves estimated time
for the pending baseline as well as the configured final margin. Completed training
calibrates the preflight timing estimate. An admitted job finishes its full schedule;
an unexpected slowdown can still exceed the deadline. Logs report measured elapsed
time and estimated remaining time rather than promising an exact finish.

Each dataset/seed job applies the budget to its own comparison. When `DATASETS`
or `SEEDS` contains several values, admission does not reserve time for all later
jobs in the outer Python loop. A shared timestamp therefore does not guarantee that
every requested comparison will fit before that time. The supplied one-dataset,
one-seed recipe has only one such comparison.

## Resume and legacy material

Resume with the same output directory and unchanged resolved experiment inputs:

```bash
python -u classification/experiments/erm_cue.py \
  --runs-dir runs/cifar100_erm_cue_trial01 \
  --resume
```

Completed artifacts are verified and reused. A changed recipe, search grid,
deadline, runtime source, or data identity requires a new run directory; do not
rewrite saved contracts to make them appear compatible. Keep a running server
process on its original source revision until it stops.

The former `classification/overnight.py` and its fixed settings/workflow were moved
to [legacy/classification_overnight](../legacy/classification_overnight/README.md)
with byte-preserving provenance. They are reference material and are not imported
by active experiments. The existing fixed `cue`/`erm`, multi-dataset ablation, full
study, PACS, and Office-Home commands remain available for their separate workflows.
