# Fixed overnight runner snapshot

These six files preserve the one-off CIFAR-100 overnight workflow from commit
`943ddeca3e21280942f778b56a11ebbae86c2751`. `SOURCE_MANIFEST.json` records each
original relative path and SHA-256; archived files retain their original bytes.

The old driver fixed the common training recipe and encoded the tuning/baseline
order inside its workflow. New experiments use
[classification/experiments/erm_cue.py](../../classification/experiments/erm_cue.py)
and the reusable [Comparison API](../../classification/EXPERIMENTS.md), where the
experiment script supplies hyperparameters, search candidates, stage order, seeds,
and optional deadline.

This is a historical reference snapshot, **not a standalone runnable release**.
The archived Python files depend on shared modules at their original revision, and
their old root-relative paths are intentionally preserved. Do not run them against
the current training engine or collect their tests with the active test suite.
Active code must not import from this directory.

To reproduce or resume a historical run, create a separate checkout/worktree of
the full original commit, use its original environment and command, and retain its
source/data/configuration contracts. Do not edit a checkpoint contract to bypass
source checks. Existing run artifacts have not been moved or changed.

The old Markdown instructions describe the old revision. Their dates, paths,
defaults, and implementation claims do not describe the current interface.
