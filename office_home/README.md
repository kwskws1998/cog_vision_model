# Office-Home: matched ResNet-18 experiments

This directory adds Office-Home to `cog_vision_model` without editing any existing PACS executable, configuration, reference, environment, or test file. The shared PACS fingerprint remains `583fc84119e4d5ff16b323e91d446f394bd824eb1f400a94ee8bfdadff9ad4e2`. Existing PACS checkpoints keep their source contract. Office-Home binds both that core and this extension in its own fingerprint.

The experiment compares ERM, the same pinned pooled-feature RSC variant, and cue credit routing on all four held-out domains, with seeds 0, 1, and 2. It uses an ImageNet-pretrained ResNet-18 with **65 outputs**, 224px input, batch 48, FP16 encoder/FP32 readout, 5,000 completed Adam updates, source-only checkpoint selection, and the existing bounded AMP overflow recovery. No method gets an extra encoder or extra trainable cue head. There are 11,209,857 trainable parameters per arm.

## Existing GPU server

Run these commands from `/workspace/cog_vision_model`. Reuse the existing `.conda` environment. The `prepare` stage checks both test suites, downloads and verifies the original-resolution dataset, and checks real-source CPU updates for all three methods. It stops immediately if any command fails.

```text
conda run --no-capture-output --prefix ./.conda python office_home/run.py prepare
```

On a single shared GPU, finish PACS before starting the Office-Home GPU stage. The `train` stage first measures all three arms at the full batch/evaluation shapes, then performs ten pretrained real-source CUDA updates per arm, then starts the matrix only if both checks pass:

```text
conda run --no-capture-output --prefix ./.conda python office_home/run.py train
```

This performs **36 sequential runs**: four target domains × three seeds × three methods. Repeating the same command performs preflight again, resumes unfinished runs from their last validation checkpoint, and verifies/reuses completed runs. All training finishes before the first target evaluation. Reports are `runs/officehome_resnet18/comparison.md` and `comparison.json`. The training log includes elapsed seconds and a per-run ETA estimate.

For a fresh machine only, create the same Conda environment before either stage with `python3 bootstrap.py --device cuda`. There are no new runtime dependencies. Do not change an environment while another experiment is using it.

## Individual commands

```text
conda run --no-capture-output --prefix ./.conda python office_home/experiment.py check
conda run --no-capture-output --prefix ./.conda python office_home/experiment.py prepare --download
conda run --no-capture-output --prefix ./.conda python office_home/experiment.py data-smoke
conda run --no-capture-output --prefix ./.conda python office_home/experiment.py profile
conda run --no-capture-output --prefix ./.conda python office_home/experiment.py data-smoke --device cuda --target Art --seed 0 --steps 10
conda run --no-capture-output --prefix ./.conda python office_home/experiment.py matrix --seeds 0 1 2 --evaluate
conda run --no-capture-output --prefix ./.conda python office_home/experiment.py report --runs-dir runs/officehome_resnet18
```

Each command must succeed before the next. The Python `run.py` stages enforce this ordering automatically. `profile` exits 2 when CUDA is unavailable and exits 1 if an arm fails the memory budget. No implicit batch-size, precision, or resolution fallback is permitted.

To train one arm, use `office_home/experiment.py train --method cue --target Art --seed 0 --run-dir runs/officehome_pilot/Art_seed0_cue`. Add `--resume` after interruption. Final target evaluation requires a completed sealed selection and is explicit: `office_home/experiment.py evaluate --run-dir ...`. Viewing exploratory target results changes the interpretation of later experiments that reuse those targets.

## Data and split policy

The default downloads four original-resolution archives from the Transfer-Learning-Library mirror. Each archive and the full canonical image collection have locally verified SHA-256 values in `references/dataset_sources.json`. Original-mode imports must match those image bytes, including manually provided directories. All 15,588 images are fully RGB-decoded and hashed. The official and DomainBed Google Drive ZIP links were unavailable on the development host; their failure did not trigger a silent resized-data fallback.

| Domain | All images / target test | Source train | Source validation |
|---|---:|---:|---:|
| Art | 2427 | 1962 | 465 |
| Clipart | 4365 | 3519 | 846 |
| Product | 4439 | 3579 | 860 |
| Real_World | 4357 | 3511 | 846 |

Within each domain/class, names are ranked by SHA-256 of `(split_seed=0, domain, class, canonical image name)`. The first `max(1, floor(n/5))` images form source validation; the rest form source training. These fixed source partitions are shared by every method and training seed. Target files are read during integrity preparation; model predictions and checkpoint selection never use that domain before final evaluation. This is our prespecified stratified source split, not an official published split or an exact DomainBed reproduction.

For an existing original archive or image directory:

```text
conda run --no-capture-output --prefix ./.conda python office_home/experiment.py prepare --archive /absolute/path/to/OfficeHomeDataset_10072016.zip
conda run --no-capture-output --prefix ./.conda python office_home/experiment.py prepare --data-root /absolute/path/to/OfficeHome
```

Choose one import path. Four TGZ files can also be supplied together after `--archive`. Extraction refuses to replace an existing image directory. Repeating `prepare --download` on an existing root re-verifies it without overwriting images.

The importer recognizes the documented Dassl train/val layout, but that archive contains 256×256 square images. Its manifest is tagged `resized_256_square`; the default `original_resolution` configuration rejects it. Using that variant requires an explicit copied configuration with a matching `data_variant` and a separate run directory. It cannot be mixed with the default experiment. Class mappings, image paths, variant, split policy, and content fingerprints are bound to each run.

## Verification and packaging

`check` runs the existing 53 PACS tests and the Office-Home tests, including exact CPU resume for all three methods, the final class label 64, target exclusion, matrix ordering, artifact matching, archive guards, and failed-preflight stopping. The real-data CPU smoke is a path check, not a target-accuracy experiment. CUDA memory, AMP stability, and runtime for this extension must be measured on the target GPU.

Generated Office-Home reports live in `office_home/verification/` and are Git-ignored so GPU runs do not create report-file pull conflicts. Release bundles include the actual reports. With `requirements-dev.txt` installed in the same Conda environment, run `python office_home/audit.py --lint` after generating the CPU reports. The audit fails on stale evidence or failed outcomes.

To make an additive ZIP, run `python office_home/bundle.py --addon-only --output /absolute/path/to/officehome_addon.zip`. Extract that ZIP inside the existing repository. A full source ZIP is produced by omitting `--addon-only`. Neither bundle includes datasets, environments, caches, or training checkpoints. The existing root `bundle.py` remains the unchanged PACS packager.
