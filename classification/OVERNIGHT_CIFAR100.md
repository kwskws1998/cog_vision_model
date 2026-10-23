# One-seed PreAct ERM and CUE validation search

This command runs CIFAR-100 only, using one training seed (0) for ERM and every CUE
candidate. It never launches a second development seed, another dataset, a learning
rate search, or corruption evaluations. It reuses the selected CUE checkpoint rather
than training it again.

The order is: all CUE candidate preflights, short ERM preflight, full CUE tuning,
validation selection, full ERM training, then final test evaluation of both models.
CUE tuning always precedes the full ERM run.

## Data boundaries

| Partition | Source | Count | Permitted use |
|---|---|---:|---|
| Train | Official training set, 450 examples/class | 45,000 | Gradient updates |
| Validation | Official training set, remaining 50 examples/class | 5,000 | Epoch logging and CUE parameter selection |
| Test | Official test set | 10,000 | ERM and selected CUE evaluation after search closure |

The existing deterministic stratified split is preserved. Source-qualified sample
identities must be disjoint. Training/validation datasets now map only their required
source arrays. This runner defers test-array verification/loading until the search
is frozen; dataset preparation can still read released files to construct the three
partitions. Initial plan output is a declared recipe, while the `[split]` line and
`plan.json` contain audited manifest counts. A malformed manifest fails before training.

The training log explicitly says `val_loss`, `val_accuracy`, and `val_n`. Validation
is never renamed test. Candidates have phase `development`, which the existing test
API rejects. After every admitted candidate completes, `selection.json` freezes the
winner using final-epoch validation accuracy, then validation NLL, then registered
order. An immutable copy of that final checkpoint is promoted for evaluation, with
the source completion hash and model hash recorded. Both initial model weights and
all common recipe fields must match ERM before the two final tests are authorized.
No best-test selection, test-driven stopping, or retraining after test access occurs.

## Paper settings and explicit differences

The ECCV 2022 Zipf's LS paper, section 4.1, states:

> SGD optimizer with 0.9 momentum, 0.1 initial learning rate, 1e-4 weight decay,

It also states:

> All experiments use MSRA initialization

[Paper, p.9](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710104.pdf#page=9)

The preset uses PreAct ResNet-18, 32x32 input, SGD, batch 128, 200 epochs, LR 0.1,
momentum 0.9, weight decay 1e-4, no Nesterov, no warmup, no clipping, and FP32
training/evaluation with TF32 disabled. LR is multiplied by 0.1 after completing
epochs 100 and 150. All registered trials share these settings.

`initialization=msra` explicitly applies Kaiming-normal fan-out/ReLU initialization
to convolutions, with BatchNorm scale 1 and bias 0; the classifier retains its
constructor initialization. This is an explicit interpretation of the paper's MSRA
statement, not proof of its unpublished exact initialization choices. The public
CIFAR constructor uses PyTorch defaults instead. Existing experiments keep
`initialization=reference` unless explicitly changed.

The author's [pinned CIFAR backbone](https://github.com/megvii-research/zipfls/blob/108e27aaf81aee0dc6dcd965ed6389f2ea06ba10/pytorch_zipfls/cifar_models/resnet.py)
defines the PreAct block/stem already implemented in this project. The new
`normalization=zipfls` option uses the author's [CIFAR preprocessing constants](https://github.com/megvii-research/zipfls/blob/108e27aaf81aee0dc6dcd965ed6389f2ea06ba10/pytorch_zipfls/datasets.py):
mean `(0.4914, 0.4822, 0.4465)`, std `(0.2023, 0.1994, 0.2010)`. Training uses
zero-padded random 32px crops and horizontal flips. Validation/test have no random
augmentation. The existing keyed augmentation and sample order give paired arms
the same images and transformations without consuming model RNG state.

This is **not an exact reproduction of the published accuracy**: holding out 5,000
training images differs from using all 50,000; there is one seed; the explicit
initialization follows paper prose rather than the public constructor; and the public
PyTorch training loop has an LR timing offset. Historical software, exact random
seeds, and kernels are not reproduced. These differences are persisted in the plan.
The direct scientific comparison here is matched ERM versus validation-selected CUE.

## Time budget and tuning

The registered ordered grid is:

1. tau 0.15, floor 0.5 (original default)
2. tau 0.05, floor 0.5
3. tau 0.30, floor 0.5
4. tau 0.15, floor 0.0
5. tau 0.05, floor 0.0
6. tau 0.30, floor 0.0

Gate temperature stays fixed at 0.05 in this search. `--tau-candidates`,
`--joint-floor-candidates`, and `--gate-temperature` provide explicit overrides;
the JSON preset can also specify any finite ordered list of CUE-only parameter
triples. All candidates use the same single training seed. Hyperparameter trials
are distinct training jobs, not additional seeds.

CUDA preflight checks **every registered CUE parameter combination first**, using
10 accepted GPU updates and validation batches for each, then checks ERM briefly.
An error in any candidate aborts before any full CUE or ERM training. This checks
forward/backward, numerical validity, optimizer updates, validation and memory fit;
it cannot guarantee that no later-epoch failure will occur. Each candidate profile
is stored separately under `preflight/`, with its exact config and source identity.

Before starting the first CUE run, the runner requires an estimate that fits that
candidate **plus the full ERM run**. Every subsequent candidate admission also
reserves the full ERM estimate. Candidate-specific preflight rates estimate time;
completed CUE runs calibrate a conservative slowdown factor applied to both future
candidate estimates and the reserved ERM estimate. Admission additionally reserves
20 minutes and applies a 1.25 safety factor. A candidate predicted to exceed the
deadline is recorded as `skip_budget`; its 200 epochs are never shortened. Once
search selection is frozen, ERM runs once at its unchanged fixed recipe.

An admitted job completes its full schedule. The deadline is a conservative
estimate, not a hard kill or guaranteed completion time: sudden GPU slowdown can
still cause an overrun. `deadline_met` records the actual outcome. If no CUE trial
fits/completes, the runner fails clearly and keeps the test sealed. The tested
models do not change after viewing final test accuracy. This bounded search is not
an exhaustive search or evidence of the globally best CUE parameters.

## Server command

Stop the old run with Ctrl+C before pulling. Existing Conda dependencies suffice;
there is no ZIP and no environment rebuild. Old run checkpoints cannot be resumed
under changed source. Use a fresh run name for this experiment.
The CUE-first update also changes source/plan identity: a run created by the older
ERM-first version needs a new run directory, not `--resume` with the updated code.

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
RUN_NAME="cifar100_preact_erm_cue_fp32_b128_seed0_tune_${STAMP}"

python -u classification/overnight.py \
  --config classification/configs/overnight_cifar100.json \
  --data-root data/classification \
  --runs-dir "runs/${RUN_NAME}" \
  --seed 0 \
  --workers 4 \
  --max-vram-gib 28 \
  --tau-candidates 0.15 0.05 0.30 \
  --joint-floor-candidates 0.5 0.0 \
  --gate-temperature 0.05 \
  --finish-by '2026-10-24T10:30:00+09:00' \
  --reserve-minutes 20 \
  --time-safety-factor 1.25 \
  --logging-steps 200 \
  --progress bar \
  2>&1 | tee "results/${RUN_NAME}.log"
)
```

For a later day, explicitly change `--finish-by`; it never silently advances to
tomorrow. `--dry-run` prints the recipe without touching data or allocating CUDA.
Existing manifests/raw data are reused. Add `--download` if CIFAR is not installed.

For resume, keep the **original** runs directory, settings, and deadline, and add
`--resume`. An already admitted interrupted candidate resumes its full schedule;
subsequent new candidates still go through time admission. Completed results are
verified and reused without reopening a test evaluation or retraining.

## Results

- `plan.json`: fixed recipe, seed, grid, split counts/identity hashes and source hash.
- `search.json`: timing-only admissions and skipped candidate indices.
- `selection.json`: validation scores and frozen selected candidate.
- `development/cue*/`: train/validation artifacts; no test predictions or test access.
- `main/seed0_erm/`, `main/seed0_cue/`: final models and test logits/labels/sample IDs.
- `study_seal.json`: authorization after training and selection closure.
- `comparison.json`, `comparison.md`: final test accuracies and CUE-minus-ERM pp.
- `progress.json`: measured active-run ETA and final completion/deadline status.

One-seed differences are descriptive; no seed variance or significance claim is made.
The local synthetic CPU smoke verifies mechanics, not CUDA speed or model accuracy.
