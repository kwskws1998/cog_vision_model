# Office-Home implementation review

## Scope and controls

The extension preserves the existing PACS files and uses its tested `CueReadout`, ResNet-18, RSC/routing objectives, AMP retry operator, metrics, sampler, RNG serialization, atomic artifacts, and selection seals. Dataset-specific runner code is isolated here because editing the original runner would invalidate the currently running PACS source contract. A regression test checks the exact unchanged PACS fingerprint. Office-Home results bind both code sets.

All three methods use 65 outputs, identical pretrained and random-head initialization per seed, the same source images and augmentations, equal domain sampling, Adam settings, precision, frozen BatchNorm statistics, validation rule, and unmasked inference. Hyperparameters are copied from the common PACS recipe without target-based tuning. Each matrix is 36 runs and 180,000 completed optimizer updates.

## Reviewed boundaries and repairs

| Boundary | Validation |
|---|---|
| Dataset identity | Exact four-domain totals, all 65 class folders, complete decoding, image hashes, known source-archive hashes, a pinned canonical image collection digest for original-mode imports, and explicit image variant |
| Original/resized distinction | Google Drive links failed; an available Dassl archive was inspected and found to contain square 256px images. The verified original-resolution mirror supplies the default. A variant mismatch fails before training. |
| Class labels | One saved alphabetical class mapping, including TV ordering and label 64; the real ResNet classifier and pretrained smoke use all 65 outputs |
| Split leakage | Deterministic class-stratified source 80/20 split; disjoint train/validation sets whose union is the target-domain full list; no target loader in source training or selection |
| Archive import | ZIP/TGZ traversal, links, special files, duplicate identities/members, ambiguous roots, and existing destination replacement are rejected |
| RNG isolation | Per-image transforms seed only the CPU default generator. Using `torch.manual_seed` there could alter CUDA RNGs; the reviewed code avoids it and tests CUDA-seed exclusion and worker independence. |
| Resume | Exact CPU model and Adam states for ERM/RSC/cue; changed recipes rejected; target access prevents further training |
| Matrix | 36 training calls precede every target evaluation; completed selections are reused; unmatched initializations and reordered target partitions are rejected |
| Reporting | Domain columns are Art/Clipart/Product/Real World; metrics are recomputed from saved logits and labels; means and SDs use seed-level domain averages |
| GPU preflight | Real pretrained source batches are separate from random-weight memory probes; the Python driver stops on a failed probe rather than continuing into training |
| Git updates | Fresh reports are ignored inside this new directory; existing PACS files and tracked reports are untouched |

The source split is a deliberate new protocol, not a claim to reproduce a published table. The pooled-feature DomainBed RSC variant remains the one named competitor; it is not the original spatial-plus-channel RSC experiment. Improvements must be established by actual held-out-domain results, not implementation tests.

## Evidence and limits

The development host has a Conda CPU environment and no CUDA device. The unit suite and real-image preparation/smokes verify implementation paths. The default original-resolution package contains all 15,588 images and its artifact identity is recorded in the preparation report. No full Office-Home matrix or target-accuracy result is generated during implementation.

The 24GB target is supported by a 20GiB process-reservation guard and dedicated profile command at batch 48 / evaluation batch 64. This host cannot certify physical GPU fit, CUDA resume equivalence, or full training stability. Run the gated GPU stage on the server after PACS finishes.

The local verification summary and source fingerprints are recorded in `verification/audit.json` and the machine-generated reports beside it. Checks are repeated against the final canonical checkout before packaging. Passed CPU tests do not imply a recognition improvement or human cognitive-model validity.
