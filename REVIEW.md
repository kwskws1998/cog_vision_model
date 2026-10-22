# Full implementation review — 23 October 2026

The environment now uses Conda. This review covered every executable source file, the experiment configuration, dependencies, official split provenance, mathematical operator, training/evaluation lifecycle, tests, documentation, and portable bundle. Findings below were repaired in the canonical project. This is a self-review backed by regression tests and runtime checks, not an independent external review or evidence of classification superiority.

## Findings and repairs

| ID | Priority | Defect or gap before review | Repair | Regression evidence |
|---|---|---|---|---|
| R01 | P1 | Repeating a completed matrix called `train(..., resume=True)` for runs that had already accessed targets. The training guard then rejected the advertised repeat operation. | Verify and reuse sealed selections; do not call training for completed runs. Verify/reuse existing target artifacts. | `test_completed_evaluated_matrix_is_read_only_on_repeat` |
| R02 | P1 | Initialization mismatches and dataset changes could be discovered only by reporting after target evaluation. The matrix plan did not bind the dataset ID. | Bind dataset/code/config before work, compare existing runs before training, and validate all completed arms and initial weights again before any target evaluation. | `test_unmatched_initialization_rejected_before_target`, `test_changed_dataset_rejected_before_any_work` |
| R03 | P2 | Final-checkpoint recovery could retain an older interrupted summary. A crash before the first checkpoint also had no supported restart path. | Commit final weights and complete summary before the seal; reconstruct missing/stale summaries and history from checkpoints; restart deterministically if only the initial contract was committed. | `test_recover_completed_checkpoint_before_final_seal`, `test_crash_before_first_checkpoint_restarts_deterministically` |
| R04 | P2 | PID-based stale-lock deletion had a race between contenders and could leave a malformed lock after a crash during lock-file creation. | Use an OS-backed `filelock` lock; retain the lock inode and release it on process death. Apply a writer lock to the matrix as well. | `test_writer_lock_exclusion_and_crash_recovery` |
| R05 | P2 | The surrogate was evaluated as `(sg(z)+u)-sg(u)`. FP32 cancellation could change its forward value and therefore the supposedly joint gradient. | Evaluate `sg(z)+(u-sg(u))`, keeping the forward correction exactly zero for finite logits. | `test_large_opposing_logits_preserve_joint_surrogate`; `verification/review_regressions.json` |
| R06 | P2 | A generic `torch==2.8.0` requirement could leave an existing CPU build installed when requesting CUDA. The previous setup used the wrong environment manager for the user's preference. | Conda Python 3.11; explicit Linux `+cu126`/`+cpu` wheel tags; compiled-runtime and module-location checks; refuse project, unrelated, and Conda base prefixes. | `test_explicit_build_tags`, `test_prefix_guards`; actual Conda installation and dependency checks |
| R07 | P2 | Validation/final-test batches did not enforce the memory budget themselves. | Enforce the same peak-reservation guard on every evaluation batch and save final-test memory statistics. | `test_memory_budget_violation_is_explicit`, `test_evaluation_enforces_memory_guard` |
| R08 | P2 | RNG capture requested states for every visible CUDA device, potentially initializing devices outside the single-GPU experiment. | Capture/restore only the active CUDA device; CPU captures contain no CUDA state. | `test_python_numpy_and_torch_rng_restore`; active-device CUDA path inspected but not executed here |
| R09 | P2 | Reports verified prediction-file hashes and metrics but did not bind the ordered prediction paths/labels to the exact held-out partition. | Record target count and ordered-partition digest in the training contract; verify both when reporting. | `test_prediction_paths_must_match_target_partition` |
| R10 | P2 | The feature-grid check used an incorrect image-size relationship. Invalid ViT grid dimensions and malformed direct gate inputs could fail late or be accepted. | Validate the actual ResNet stride-32 feature extent, positive integer token grids, normalized finite FP32/FP64 cue inputs, and configuration types/ranges. | Invalid-grid/config/gate tests in `test_protocol.py` and `test_review.py` |
| R11 | P2 | Dataset verification rejected symlinked files but could follow a symlinked class/domain directory. Noncanonical path aliases were accepted. | Check every path component and require canonical relative names, alongside the existing full decoding and content hashes. | `test_noncanonical_paths_and_symlinked_class_rejected` |
| R12 | P2 | The evidence audit checked existence/fingerprints but could accept a failed smoke result carrying matching metadata. | Validate successful method outcomes, finite losses, matched controls, Conda environment identity, real-data counts, and setup/recipe metadata hashes. | `test_audit_rejects_failed_smoke_with_matching_metadata`; final `audit.json` |
| R13 | P3 | User-supplied seed order could conflict with the reporter's canonical ordering. The bundle did not include YAML recipes. Sources had unused imports and long dense statements. | Canonicalize unique seeds; include `environment.yml`; format and statically check all project Python files; migrate every executable instruction to Conda. | `test_seed_input_order_is_canonical`; static checks and final ZIP inspection |

P1 findings affected execution or whether an unmatched comparison could reach target data. P2 findings affected numerical correctness, recovery, integrity, resource boundaries, or input validation. P3 findings affected reproducibility and delivery.

## Reproduced numerical defect

The original portable bundle was preserved outside the delivery package for comparison. Its actual `routed_loss` implementation was loaded for a controlled FP32 probe, rather than reconstructing an approximation of the old function. The two cue vectors were `[1e8, 1e8, 0]` and `[-1e8+8, -1e8, 0]`, with equal areas, class label 2, and joint gates set to one. The joint logits were `[4, 0, 0]`.

The maximum error against the analytical joint gradient was **0.3156648874 before review**, and **0.0 after the fix** on this CPU environment. This is a numerical implementation counterexample, not a representative PACS sample or an efficacy result. The test is intentionally extreme so that the floating-point failure is unambiguous. Ordinary-scale analytical-gradient, all-joint ERM, gauge-invariance, and single-cue checks also remain in the suite.

## Coverage from setup through reporting

| Area | What was inspected and exercised |
|---|---|
| Setup and packaging | Conda prefix isolation, Linux wheel tags, CPU installation, actual package origins/runtime, dependency consistency, native and pip exports, YAML inclusion, excluded datasets/caches/environments, ZIP CRCs |
| Configuration and provenance | Positive dimensions, finite numeric settings, integer/bool distinction, unknown keys, supported precision/device combinations, source/config/dependency/split fingerprints, manifest and initial-weight identities |
| Data | All 9991 real PACS images; full RGB decoding; official labels/counts/partition unions; 12 pinned Git blob identities; safe archive import; symlink/path rejection; balanced sampling with replacement; augmentation isolation and workers 0 versus 2 |
| Cue operator | Area preservation on a 7×7 map, shared classifier/bias, exact per-cue gradient, detached gates, categorical disagreement bounds, independent logit shifts, single-cue and all-joint ERM limits, marginal/global CE–KL identity, cancellation stability |
| RSC | Signed correct-class logit derivatives, strict percentile ties, batch probability-drop selection, mask composition, no higher-order graph, surviving encoder gradient, NumPy reference checks in FP32 and FP64 |
| Backbone and inference | ImageNet weight loading, 11,180,103 trainable parameters, identical initial states across arms, frozen BN running statistics with trainable affine weights, unmasked common inference, explicit patch-token grids |
| Optimization and resources | Finite losses/gradients, scaler/unscale/step ordering, matching Adam settings, optional clipping, no silent batch/resolution fallback, materialized optimizer-state memory probes, guard propagation during evaluation |
| Checkpoints and selection | Atomic checkpoints, source-only equal-domain validation, tie rules, saved selected weights, exact CPU optimizer/model resume, RNG restoration, interrupted initial/final writes, kernel writer locks, source/data/runtime mismatch rejection |
| Matrix and final testing | Sorted unique seeds, bound dataset plan, pre-target initialization matching, all-training-before-target ordering, completed-run reuse, sealed target access, repeated-access journal, refusal to continue training after target access |
| Metrics and evidence | Target-path/label partition binding, prediction hashes, metric recomputation, equal-domain averages, SD across seed-level means, paired cue-minus-RSC differences, debug exclusion, successful-outcome checks and source-current reports |

## Mathematical and experimental scope

The review preserved the proposed update in METHOD.md. The CNN readout still adds zero trainable parameters and collapses to the same global-average linear classifier at inference. Region areas remain unequal where needed. Gates remain detached and cue specific. RSC remains the explicitly documented pooled-feature DomainBed adaptation, with the same common training recipe as the other arms. No result is substituted from published tables.

Passing these checks establishes code-path behavior and the tested identities. It does not establish that spatial regions are independent cognitive cues, that the update is novel, that it minimizes a conventional global smooth objective, or that it improves domain generalization. Those claims remain outside the implementation evidence.

## Remaining limits

- **CUDA is unavailable locally.** Actual 24 GB fit, CUDA FP16/BF16 execution, scaler behavior on the target GPU, CUDA resume equivalence, and Linux CUDA installation still require the target machine. Mocked memory-guard tests verify control flow, not physical VRAM use.
- **The complete 36-run benchmark has not run.** Real-PACS CPU checks execute small source-only optimizer steps and limited source-validation forwards; they provide no held-out-domain accuracy conclusion.
- **ViT support is at the readout interface.** A complete pretrained ViT training recipe and comparison are not implemented. Mean patch pooling must be matched across controls; it is not equivalent to retaining a CLS-token head.
- Native Conda explicit exports are platform specific. Use `environment.yml` plus `bootstrap.py --device cuda` on Linux; do not treat the macOS CPU export as a portable CUDA lock file.
- Dataset import deliberately refuses to overwrite an existing image root. Use `prepare --data-root ...` to recheck already extracted images. Exact resume requires unchanged code, configuration, dataset, initial weights, and runtime; the reviewed schema is not an automatic migration path for older run artifacts.

No identified CPU-path defect was deliberately left unfixed. GPU execution and scientific efficacy remain unverified, and further review can still uncover additional issues.
