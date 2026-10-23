# CIFAR-100 CE reference reproduction

This run isolates the CE baseline from Liang et al., *Efficient One Pass
Self-distillation with Zipf's Label Smoothing* (ECCV 2022). It is not a CUE run.
The reference target is Table 2's CIFAR-100 ResNet-18 **75.51 +/- 0.28%** over five runs.
One seed is a diagnostic reproduction attempt, not a reproduction of that five-run statistic.

Paper evidence (Table 2):

> Vanilla 77.86±0.26 60.31±0.36 75.51±0.28 56.41±0.20

Paper evidence (section 4.1):

> SGD optimizer with 0.9 momentum, 0.1 initial learning rate, 1e-4 weight decay,

[Paper, p.9](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710104.pdf#page=9).
The first pair in the quoted row is DenseNet-121; the last pair is ResNet-18.

## Protocol and discrepancies

The default follows the [author's PyTorch CE baseline](https://github.com/megvii-research/zipfls/blob/108e27aaf81aee0dc6dcd965ed6389f2ea06ba10/pytorch_zipfls/README.md),
at pinned revision `108e27aaf81aee0dc6dcd965ed6389f2ea06ba10`.
The original backbone file is hash-checked and executed directly. It instantiates
the same unused auxiliary heads, consumes the same constructor random draws,
and uses the author's final-head forward computation. Only final-head CE is differentiated.

| Setting | Default |
|---|---|
| Dataset | CIFAR-100, fine labels |
| Training set | All 50,000 official training examples |
| Evaluation set | All 10,000 official test examples |
| Backbone | Author CIFAR_ResNet18: PreActBlock, [2,2,2,2], 32px stem |
| Objective | Final-head CE, no smoothing, no auxiliary CE, no distillation |
| Optimizer | SGD, momentum 0.9, weight decay 1e-4, Nesterov off |
| Train / evaluation batch | 128 / 128, no dropped final batch |
| Epochs / learning rate | 200 / 0.1, decay factor 0.1, no warmup |
| Precision | FP32 training and evaluation; TF32 explicitly disabled |
| Train transforms | RandomCrop(32, padding=4, zero padding), RandomHorizontalFlip, ToTensor, Normalize |
| Test transforms | Resize(32), ToTensor, Normalize |
| Mean / std | (0.4914,0.4822,0.4465) / (0.2023,0.1994,0.2010) |
| Seed / workers | 0 / 4; override seeds to run more independent jobs |

Three details must not be silently described as identical to the paper prose:

1. **LR timing.** The paper names 100 and 150. The published `train.py` invokes
   `adjust_learning_rate` *after* training and evaluation, with a zero-based epoch.
   Its `utils.py` tests `epoch >= 100` / `>= 150`. Thus the actual first reduced-LR
   training epochs are **102 and 152**, one-based. `--schedule official-code`
   preserves this behavior. `--schedule paper` instead changes LR after completing
   epochs 100 and 150 (first used at 101 and 151). These are different labeled runs.
2. **Initialization.** The paper describes MSRA initialization, but the pinned
   `CIFAR_ResNet` constructor does not explicitly reinitialize its Conv/Linear layers.
   The runner preserves that constructor, not the different initialization loop in
   the same source file's ImageNet `ResNet` class. Do not call this an exact
   implementation of every statement in the paper. `--schedule paper` changes only
   LR timing, not initialization.
3. **Evaluation / aggregation.** The author's `datasets.py` names its loader `val`
   but loads `CIFAR100(train=False)`: the official test set. Its training loop evaluates
   it every epoch; `gen_results_sheet.py` takes the maximum reported accuracy across
   epochs. We expose **final_test** and **best_observed_test** separately, and retain
   the fixed final epoch as the primary result. Best observed test accuracy is
   test-selected and is not an independent validation-selected estimate. The public
   aggregation code is evidence of code behavior, not proof of the exact checkpoints
   behind every published table entry.

The pinned code does not specify the five paper seed values or fully pin its original
environment. This runner adds explicit RNG seeds, records the current environment,
disables TF32, and saves complete restart checkpoints. It does not promise identical
historical CUDA numerics or the published accuracy. Changing the batch to 1024 to
fill VRAM would change this baseline experiment; keep 128 for this run.

## Server execution

Stop or finish an older training process before updating its source checkout.
The ordinary study binds checkpoints to its source hash, so updating this checkout
does not migrate an in-progress old study to the new version.

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
RUN_NAME="eccv2022_author_ce_preact_cifar100_fp32_b128_seed0_${STAMP}"

python -u classification/reproduce_baseline.py \
  --config classification/configs/eccv2022_cifar100_baseline.json \
  --data-root data/classification \
  --runs-dir "runs/${RUN_NAME}" \
  --seeds 0 \
  --epochs 200 \
  --batch-size 128 \
  --eval-batch-size 128 \
  --learning-rate 0.1 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --milestones 100 150 \
  --lr-decay 0.1 \
  --schedule official-code \
  --device cuda \
  --workers 4 \
  --logging-steps 200 \
  --progress bar \
  2>&1 | tee "results/${RUN_NAME}.log"
)
```

Existing project CIFAR files are reused, including `raw/cifar100/checked`.
Use `--download` only if official raw files are absent. No additional requirements
are introduced. If `git pull` reports local modifications, preserve those changes
and resolve them before continuing; this command does not reset or stash them.

Use `--dry-run` to inspect the recipe without loading data or allocating a GPU.
For resume, repeat the same command with the **original** run directory and
`--resume`. Do not generate a new timestamp. It resumes from the last completed epoch,
and rejects changes in recipe, data, reference code, runtime environment or runner source.
No checkpoint is reused from the old 45k/5k CUE study.

Each epoch prints train/test loss and accuracy, throughput, peak reserved VRAM,
elapsed time, remaining time, and estimated completion in KST. ETA becomes available
after the first full epoch; it includes measured training and test evaluation.
At the default setting there are **391 train batches per epoch**, **79 test batches**,
and **78,200 optimizer steps** over 200 epochs. No reliable 5090 duration has been
measured for this FP32 reference path yet.

## Outputs and verification

- `results.md`, `results.json`: final test and best observed test, separately labeled.
- `seed0/contract.json`: recipe, raw data hashes, source hashes, reference provenance, environment.
- `seed0/history.json`, `status.json`: measured progress and all epoch metrics.
- `seed0/last.pt`: model, SGD momentum, RNG, history; atomic epoch checkpoint.
- `seed0/final.pt`: final-epoch weights, independent of the best test epoch.
- `seed0/test_predictions.pt`: final logits, labels, ordered official test sample IDs.
- `seed0/summary.json`: final metrics, checkpoint/prediction hashes, elapsed time.

The final checkpoint state is evaluated again after training; its count and loss
must agree with the final epoch. Metrics are recomputable from saved logits/labels.
CPU tests compare LR and transforms with the independently executed pinned functions,
compare CE/SGD weight updates, reject split/count mistakes, and check bitwise CPU resume.

```bash
python classification/reproduce_baseline.py --smoke \
  --runs-dir /tmp/eccv2022_ce_smoke --progress plain
```

The smoke run is synthetic and marked accordingly; it is not evidence of benchmark accuracy.

## Planned external comparison

Elliptic Loss Regularization (ICLR 2025) is the selected next external comparison
target, especially for TinyImageNet 64px, where it directly reports PreAct ResNet-18.
Its CIFAR-100 table uses deeper backbones, so a matched PreAct-18 experiment is still
required. Its implementation and training cost have not yet been validated here;
this baseline change does not implement or launch Elliptic.

Exact Table 2 fragment:

> Elliptic 65.03±0.1 84.98±0.1

[Paper](https://arxiv.org/html/2503.02138v1). This is an external reported result,
not a matched result against our CUE runs and not a claim of being the newest method available.
