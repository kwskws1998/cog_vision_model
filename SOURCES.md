# Direct evidence and provenance

The excerpts below are word-for-word evidence. The proposed vision operator and its hyperparameters are our design choices, not equations attributed to these papers.

## Human modelling source

Qingtian Mi and Christopher Summerfield, **Human curriculum learning of a cue combination task**, Nature Human Behaviour (published 5 May 2026).

Source: https://www.nature.com/articles/s41562-026-02452-1

Exact abstract excerpt:

> a hybrid learning framework that arbitrates between two different learning strategies

Exact Methods, Model 4 excerpt:

> distance between the two error signals

The paper distinguishes marginal credit assignment from joint credit assignment. Its hybrid PE model arbitrates using prediction-error discrepancy; its hybrid s.d. model uses variability of learned cue weights as a heuristic. The primary fitted-model discussion favors the s.d. heuristic's cognitive plausibility. The current implementation is a categorical adaptation inspired by the PE discrepancy idea. It uses smooth cue-specific detached gates, a joint floor, and spatial feature regions. Those are explicit departures from the human model. It does not instantiate fitted human parameters, reproduce behavioral fits, or validate a resource-rational account of CNN computation.

## Selected competitor and task

Zeyi Huang, Haohan Wang, Eric P. Xing, and Dong Huang, **Self-Challenging Improves Cross-Domain Generalization**, ECCV 2020.

Paper: https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123470120.pdf

Exact abstract excerpt:

> the dominant features activated on the training data

Exact PACS protocol excerpt, Section 4.1:

> train a model on three domains

RSC is a direct competitor for encouraging use of alternative visual evidence during supervised training. PACS gives a defined held-out visual-domain outcome, and the paper explicitly evaluates ResNet-18. This motivates the choice; it does not establish RSC as the strongest available 2026 baseline. The operator here follows the pooled-feature DomainBed variant, not the paper's spatial-plus-channel experiment or its reported scores.

Pinned operator implementation:

- Repository: https://github.com/facebookresearch/DomainBed
- Commit: `b93c22a1cfc3b2428398272c1a116c8de1f4139e`
- Code: https://github.com/facebookresearch/DomainBed/blob/b93c22a1cfc3b2428398272c1a116c8de1f4139e/domainbed/algorithms.py
- Verbatim reference: `references/domainbed_rsc_reference.txt`
- License: `references/DOMAINBED_LICENSE.txt` (MIT)

Pinned official RSC split source:

- Repository: https://github.com/DeLightCMU/RSC
- Commit: `bf6d280c5d74910f009ea8963c59167252659666`
- Split folder: https://github.com/DeLightCMU/RSC/tree/bf6d280c5d74910f009ea8963c59167252659666/Domain_Generalization/data/correct_txt_lists
- Included files: `references/pacs_splits/*_kfold.txt`
- License: `references/RSC_LICENSE.txt` (BSD-2-Clause)

PACS archive reference: DomainBed `domainbed/scripts/download.py` at the pinned commit. Archive URL: https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd . Archive and individual image hashes are generated on preparation rather than inventing an unverified archive checksum.

Fallback archive reference: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md (verified during implementation). Its PACS URL is https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE . This supplies image bytes only; the pinned RSC split files remain the protocol. The PACS authors' project page is https://dali-dl.github.io/project_iccv2017.html . Download availability does not establish dataset integrity; the complete manifest preparation still must succeed.

PyTorch installation compatibility: https://pytorch.org/get-started/previous-versions/ . The pinned pair is torch 2.8.0 / torchvision 0.23.0 with official CUDA 12.6 wheels. Pretraining is torchvision's `ResNet18_Weights.IMAGENET1K_V1`, from https://download.pytorch.org/models/resnet18-f37072fd.pth . The download checks the official filename hash prefix, and the resulting initial model state is fingerprinted for each run.

Conda environment management and explicit prefixes: https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html . `environment.yml` defines the native Python environment; `bootstrap.py` installs the selected official PyTorch build inside that Conda prefix and records exact installed package lists.
