# PTFuse: Prompt-Guided Third-Party Modality Exploration

<p align="center">
  <a href="https://ieeexplore.ieee.org/abstract/document/11598884">
    <img src="https://img.shields.io/badge/Paper-IOTJ-blue" alt="Paper">
  </a>
  <a href="README_CN.md">
    <img src="https://img.shields.io/badge/中文-README_CN-red" alt="Chinese README">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.4.1-ee4c2c" alt="PyTorch">
</p>

This repository provides the **official implementation** of:

> **PTFuse: Prompt-Guided Third-Party Modality Exploration for Infrared-Visible Image Fusion**

The paper is published in the **IEEE Internet of Things Journal (IOTJ)**.

- **Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11598884)
- **中文说明:** [`README_CN.md`](README_CN.md)

## Overview

PTFuse treats an infrared-visible fused image as a distinct **third-party modality**
rather than as a simple pixel- or feature-level mixture of the two source images.
The framework contains two principal components:

- **TeMAL — Text-Guided Modality-aware Adversarial Learning:** uses CLIP text
  prompts to provide global semantic specifications and modality-aware adversarial
  supervision.
- **PATS — Patch-Aware Triplet Supervision:** identifies high-information local
  patches and suppresses deceptive pseudo-fused patterns caused by spatially
  mismatched infrared-visible pairs.

The implementation is organized as follows:

```text
model/PTFuse/
├── model.py       # PTFuse training and inference pipeline
├── network.py     # PTFuseGenerator, TeMAL, and modality discriminators
└── loss.py        # Fusion loss and PATS
```

## Method at a Glance

The following figures are reproduced from the paper to summarize the proposed
architecture and its main components.

![PTFuse framework](figs/method.png)

*Overall PTFuse framework, including the prompt-guided adversarial branch and
patch-aware triplet supervision.*

![Encoder and decoder](figs/en_de.png)

*Encoder-decoder path used to construct the fused third-party modality.*

![Misaligned fusion patterns](figs/wrong_patterns.png)

*PATS motivation: spatially misaligned source pairs can produce deceptive
pseudo-fused patterns, which are suppressed by patch-aware supervision.*

![TeMAL design](figs/idea1.png)

*TeMAL introduces text-guided modality awareness into adversarial learning.*

![PATS design](figs/idea2.png)

*PATS identifies and filters locally incompatible third-party modality patterns.*

## Environment

The commands below are independent of the repository author's machine. Python
3.12 is recommended; use an isolated environment rather than a global
installation:

```bash
conda create -n ptfuse python=3.12 -y
conda activate ptfuse
```

Install the important runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

For NVIDIA GPUs, install the PyTorch build matching the driver from the
[official selector](https://pytorch.org/get-started/locally/) before installing
the remaining requirements. The following versions were used for verification:

| Component | Version |
|---|---:|
| Python | 3.12.4 |
| PyTorch | 2.4.1+cu118 |
| torchvision | 0.19.1+cu118 |
| CUDA runtime | 11.8 |
| NumPy | 1.26.4 |
| OpenCV | 4.10.0 |
| Kornia | 0.8.0 |
| einops | 0.8.0 |
| CLIP | 1.0 |

The repository includes the important Python dependencies in
[`requirements.txt`](requirements.txt). CPU execution is supported, but CLIP
training and inference are substantially slower.

The CLIP `ViT-B/32` weights are downloaded on first use. Internet access or a
pre-populated CLIP cache is therefore required. The vector figure originals and
GitHub-renderable previews are available in [`figs/`](figs/).

## Datasets

The paper evaluates PTFuse on the following public infrared-visible datasets:

| Dataset | Protocol |
|---|---|
| MSRS | Training and in-domain testing; 361 test pairs are reported in the paper |
| FMB | Cross-dataset testing; 280 pairs are reported in the paper |
| M3FD | Cross-dataset testing; 300 pairs are reported in the paper |

For the paper protocol, the dataset set is:

```python
choices = ["MSRS", "FMB", "M3FD"]
```

This describes the datasets used in the paper; it does not imply that every
dataset loader is already implemented in `libs/data.py`. Check the loader before
running a new dataset.

A typical MSRS layout is:

```text
data/
└── MSRS/
    ├── train/
    │   ├── ir/
    │   └── vi/
    └── test/
        ├── ir/
        └── vi/
```

The exact directory names are determined by [`libs/data.py`](libs/data.py).

The prompt option accepts `discussion`; omit it or pass `None` to disable text
prompts. The corresponding `discussion.txt` file must be placed in each dataset
split directory when text guidance is enabled.

## Training

The paper uses 96x96 random crops, batch size 2, 100 epochs, and AdamW with a
learning rate of `1e-4`. PATS uses patch size 32, `K_pos=2`, `K_neg=8`, and
temperature `0.1`.

Run from the repository root:

```bash
python main.py PTFuse \
  --data MSRS \
  --phase train \
  --batch_size 2 \
  --epochs 100 \
  --aug_methods random_crop RVF RHF \
  --convert_mode RGB \
  --prompt discussion \
  --control_save \
  --control_save_weights \
  --control_save_img
```

PATS requires a batch size of 2 in the current implementation because each
iteration constructs both aligned and mismatched infrared-visible pairs.

## Testing

```bash
python main.py PTFuse \
  --data MSRS \
  --phase test \
  --batch_size 2 \
  --prompt discussion \
  --control_save \
  --control_save_img \
  --convert_mode RGB
```

Before testing, provide a trained checkpoint in the default output directory
or pass its path through `--weight_path`.

## Evaluation Metrics

Metric computation is intentionally kept outside the PTFuse training and
inference pipeline. If quantitative evaluation is required, please refer to
the metric implementations and evaluation protocols in:

- [RollingPlain/IVIF_ZOO](https://github.com/RollingPlain/IVIF_ZOO)
- [Zhaozixiang1228/MMIF-CDDFuse](https://github.com/Zhaozixiang1228/MMIF-CDDFuse)

When comparing results, use the same dataset split, image conversion,
normalization, metric implementation, and official checkpoint protocol.

## Comparison Methods

The comparison methods reported in the paper are evaluated using the official
publicly released pretrained weights whenever available. Their code and weights
should be obtained from the corresponding authors or project pages:

- [LRRNET](https://ieeexplore.ieee.org/document/10105495)
- [CDDFuse](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_CDDFuse_Correlation-Driven_Dual-Branch_Feature_Decomposition_for_Multi-Modality_Image_Fusion_CVPR_2023_paper.html)
- [TGFuse](https://ieeexplore.ieee.org/document/10122870)
- [DDFM](https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_DDFM_Denoising_Diffusion_Model_for_Multi-Modality_Image_Fusion_ICCV_2023_paper.html)
- DDBF (CVPR 2024): obtain the official implementation and checkpoint from the authors' public release
- [EMMA](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Equivariant_Multi-Modality_Image_Fusion_CVPR_2024_paper.html)
- [Text-IF](https://openaccess.thecvf.com/content/CVPR2024/html/Yi_Text-IF_Leveraging_Semantic_Text_Guidance_for_Degradation-Aware_and_Interactive_Image_CVPR_2024_paper.html)
- [FreqGAN](https://ieeexplore.ieee.org/document/10680110)
- [TextFusion](https://doi.org/10.1016/j.inffus.2025.103046)
- [PromptF](https://ieeexplore.ieee.org/document/10815008)

## Results

The following values are transcribed from the paper's main comparison table.
They are included as a compact reference; reproduce or recompute metrics with
the evaluation protocols described above before making a new comparison.

| Dataset | Method | VIF | EN | SD | SF | AG | EI |
|---|---|---:|---:|---:|---:|---:|---:|
| MSRS | PTFuse | **1.0866** | **6.8006** | **44.0214** | **11.9349** | **3.9808** | **44.3087** |
| MSRS | Best competing value | 1.0775 | 6.7866 | 44.3498 | 12.5340 | 3.9229 | 43.7947 |
| FMB | PTFuse | 0.9156 | **6.8231** | **38.3918** | **15.1448** | **4.5268** | **50.5642** |
| FMB | Best competing value | **0.9353** | 6.9866 | 37.2790 | 14.9651 | 4.4930 | 49.7972 |
| M3FD | PTFuse | 0.8384 | **6.9463** | **38.3980** | **15.9305** | **5.2539** | **56.8797** |
| M3FD | Best competing value | **0.8884** | 6.8927 | 36.7622 | 15.7059 | 5.2088 | 56.1677 |

The ablation study on MSRS reports the following full-model values:

| Variant | VIF | EN | SD | SF | AG | EI |
|---|---:|---:|---:|---:|---:|---:|
| Without contrastive loss | 1.0791 | 6.7587 | 43.2963 | 11.6623 | 3.8697 | 42.9804 |
| Global PATS | 1.0820 | 6.7979 | 43.8675 | 11.8669 | 3.9299 | 43.6544 |
| Random sampling | 1.0720 | 6.7618 | 42.9084 | 11.6193 | 3.8657 | 42.8980 |
| **PTFuse** | **1.0866** | **6.8006** | **44.0214** | **11.9349** | **3.9808** | **44.3087** |

For downstream tasks, the paper reports an M3FD detection score of **0.880
mAP@50** and **0.577 mAP@50:95**. The reported semantic-segmentation mIoU is
**60.42** on FMB and **77.57** on MSRS.

### Qualitative Results

![MSRS fusion comparison](figs/fused_MSRS.png)

*Low-light fusion comparison on MSRS.*

![FMB fusion comparison](figs/fused_FMB.png)

*Cross-dataset fusion comparison on FMB.*

![M3FD fusion comparison](figs/fused_M3FD.png)

*Rainy-scene fusion comparison on M3FD.*

![M3FD detection comparison](figs/detect_M3FD.png)

*Downstream object-detection comparison on M3FD.*

![FMB segmentation comparison](figs/seg_FMB.png)

*Downstream semantic-segmentation comparison on FMB.*

![MSRS segmentation comparison](figs/seg_MSRS.png)

*Downstream semantic-segmentation comparison on MSRS.*

## Visualization

The project supports saving fused images and optional feature, attention, filter,
and monitoring visualizations. Relevant switches are exposed by `libs/opt.py`,
including `--control_print`, `--control_save`, `--control_save_img`,
`--control_save_img_type`, and `--control_monitor`.

To enable Visdom:

```bash
python -m visdom.server -p 8097
```

Then run the experiment with `--control_monitor 1`.

## Citation

If you use this code or PTFuse in your research, please cite the IOTJ paper:

```bibtex
@article{liu2026ptfuse,
  title   = {PTFuse: Prompt-Guided Third-Party Modality Exploration for Infrared-Visible Image Fusion},
  author  = {Liu, Jianpu and Yang, Yang and Lang, Yue and Miao, Di and Chen, Mo},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  url     = {https://ieeexplore.ieee.org/abstract/document/11598884}
}
```

## Acknowledgements

We thank the authors and maintainers of the comparison methods and their
public implementations, including LRRNET, CDDFuse, TGFuse, DDFM, DDBF, EMMA,
Text-IF, FreqGAN, TextFusion, and PromptF. Their released models and papers
make reproducible evaluation and fair comparison possible. Please cite the
original work when using any of these methods.
