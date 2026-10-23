# Archived experiment code

This directory preserves superseded experiment drivers and their original inputs.
Active training code does not import these archives.

- [classification_overnight](classification_overnight/README.md): the fixed-recipe,
  one-seed CIFAR-100 deadline runner from commit `943ddec`, replaced by the reusable
  [Python comparison API](../classification/EXPERIMENTS.md).

Results, datasets, checkpoints, and installed environments stay in their existing
locations. Archiving a driver does not migrate or revalidate an old result.
