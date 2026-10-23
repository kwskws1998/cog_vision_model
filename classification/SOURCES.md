# Sources and explicit adaptation boundaries

## Selected external method

Liang et al., Efficient One Pass Self-distillation with Zipf's Label Smoothing, ECCV2022.

Paper: https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136710104.pdf

Exact Figure3 excerpt:

> We apply a shared-classifier on dense feature maps

Official implementation: https://github.com/megvii-research/zipfls/tree/108e27aaf81aee0dc6dcd965ed6389f2ea06ba10

The exact files and their measured SHA-256 values appear in `references/zipfls_provenance.json`. The independent port preserves dense2 voting and auxiliary supervision; it changes the shared small-image backbone and retains native64px Tiny images. Published scores are not matched experimental evidence. The source code is Apache-licensed; its complete license is included as an inert reference text file.

## Mechanism experiment

Shah et al., The Pitfalls of Simplicity Bias in Neural Networks, NeurIPS2020.

Paper: https://proceedings.neurips.cc/paper/2020/file/6cfe0e6127fa25df2a0ef2ae1067d915-Paper.pdf

Exact abstract excerpt:

> neural networks can exclusively rely on the simplest feature

Official implementation: https://github.com/harshays/simplicitybiaspitfalls

The present95%/50%/5% pairing and counterfactual evaluation are our prespecified experimental variant. They are not attributed to the original paper's exact protocol.

## Data sources

- CIFAR-100/CIFAR-10: https://cave.cs.toronto.edu/kriz/cifar.html ; torchvision verifies the original files' MD5 checksums before conversion. Fine CIFAR-100 labels are used. The byte-identical Paddle mirror is documented in https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/dataset/cifar.py ; the complete original MD5 remains mandatory before safe extraction.
- MNIST: torchvision's checked original IDX releases and mirrors.
- Tiny ImageNet original archive: http://cs231n.stanford.edu/tiny-imagenet-200.zip . Identical-byte mirror with published MD5: https://zenodo.org/records/10720917 . Required archive MD5: `90528d7ca1a48142e341f4ef8d21d0de`.
- CIFAR-100-C author release: https://zenodo.org/records/3555552 . Required archive MD5: `11f0ed0f1191edbf9fa23466ae6021d3`.

Mirror retrieval metadata records the selected mirror URL and measured archive hash. A mirror changes transport only; mismatched archives are rejected. Dataset preparation reads final image bytes for integrity, but no final predictions, accuracy, checkpoint decisions, or gate updates.
