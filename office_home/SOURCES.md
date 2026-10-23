# Office-Home data and protocol provenance

Official dataset page: https://www.hemanthdv.org/officeHomeDataset.html

Exact dataset-page excerpt:

> around 15,500 images from 65 different categories.

Original citation: Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan, *Deep Hashing Network for Unsupervised Domain Adaptation*, CVPR 2017. The use here is held-out-domain generalization under our specified source-only selection rule.

Original-resolution mirror reference: https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/vision/datasets/officehome.py . Its four domain archive URLs and the SHA-256 values measured from successful downloads on 23 October 2026 are saved in `references/dataset_sources.json`. These measured values are not represented as author-published checksums. The full 15,588-image package was decoded and hashed during implementation.

Other checked sources:

- The official Google Drive ZIP link from the dataset page: https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view?resourcekey=0-2SNWq0CDAuWOBRRBL7ZZsw . Unavailable through gdown on the development host.
- DomainBed's archive URL, from its already-pinned `b93c22a1cfc3b2428398272c1a116c8de1f4139e` download script: https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC . Also unavailable through gdown there.
- Dassl Office-Home-DG reference: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/datasets/dg/office_home_dg.py . Its documented archive at https://drive.google.com/uc?id=1gkbf_KaxoBws-GWT3XIPZ7BnkqbAxIFa downloaded successfully. Its 15,588 images are resized to 256×256 and arranged by domain/train-or-val/class. The default experiment does not silently substitute these pixels for the original-resolution package.

The canonical class order and source split algorithm are implementation choices defined in `ohvision/config.py` and `ohvision/data.py`. The split is fixed before training and independent of all training seeds and target scores. It is not an official Office-Home DG split or an exact reproduction of DomainBed's per-trial random holdout.

The cognitive inspiration, RSC citation, pinned RSC operator, and licenses remain in the unchanged parent `SOURCES.md` and `references/`. No new cognitive-model or novelty claim is made by this dataset extension.
