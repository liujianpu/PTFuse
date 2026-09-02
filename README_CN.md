# PTFuse：面向红外-可见光图像融合的提示引导第三方模态探索

[返回英文主页](README.md)

本仓库是以下论文发表在 **IEEE Internet of Things Journal（IOTJ）** 的官方代码：

> **PTFuse: Prompt-Guided Third-Party Modality Exploration for Infrared-Visible Image Fusion**

- **论文链接：** [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11598884)
- **English README：** [`README.md`](README.md)

## 方法概述

PTFuse 将红外-可见光融合图像视为具有独立属性的 **第三方模态**，而不是
红外图像和可见光图像的简单拼接或折中。

模型包含两个核心模块：

- **TeMAL（Text-Guided Modality-aware Adversarial Learning）**
  - 使用 CLIP 文本提示提供全局语义约束；
  - 通过文本引导的对抗学习增强模态感知能力。
- **PATS（Patch-Aware Triplet Supervision）**
  - 在局部 patch 级别选择高信息区域；
  - 构造错位伪融合样本；
  - 抑制错误的第三方模态模式。

当前实现位于：

```text
model/PTFuse/
├── model.py       # PTFuse 训练和推理流程
├── network.py     # PTFuseGenerator、TeMAL 和模态判别器
└── loss.py        # Fusion loss 和 PATS
```

## 方法图示

以下图片摘自论文，用于展示 PTFuse 的整体结构及其主要模块。

![PTFuse整体框架](figs/method.png)

*PTFuse 整体框架，包括提示引导的对抗分支和面向 patch 的三元组监督。*

![编码器和解码器](figs/en_de.png)

*用于构造第三方融合模态的编码器-解码器路径。*

![错位融合模式](figs/wrong_patterns.png)

*PATS 的动机：错位的源图像对可能产生具有欺骗性的伪融合模式，
面向 patch 的监督用于抑制这类模式。*

![TeMAL模块](figs/idea1.png)

*TeMAL 将文本引导的模态感知能力引入对抗学习。*

![PATS模块](figs/idea2.png)

*PATS 用于识别和过滤局部不一致的第三方模态模式。*

## 环境

推荐使用 Python 3.10-3.12，并创建独立环境：

```bash
conda create -n ptfuse python=3.12 -y
conda activate ptfuse
```

然后安装仓库中的主要依赖：

```bash
python -m pip install -r requirements.txt
```

如果使用 NVIDIA GPU，请先根据驱动在
[PyTorch 官网](https://pytorch.org/get-started/locally/)选择对应版本，再安装
其余依赖。作者用于验证的环境版本如下：

| 组件 | 版本 |
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

仓库已在 `requirements.txt` 中列出主要运行依赖。CPU 模式可以运行，但 CLIP
训练和推理会明显变慢。

首次运行时需要下载 CLIP `ViT-B/32` 权重，请确保网络可用，或提前配置 CLIP
缓存目录。论文图的 PDF 矢量文件和 GitHub 展示用 PNG 预览图位于
[`figs/`](figs/)。

## 数据集

论文中使用三个公开红外-可见光数据集：

| 数据集 | 用途 |
|---|---|
| MSRS | 训练和同域测试，论文报告测试集包含 361 对图像 |
| FMB | 跨数据集测试，论文报告 280 对图像 |
| M3FD | 跨数据集测试，论文报告 300 对图像 |

论文实验对应的数据集集合为：

```python
choices = ["MSRS", "FMB", "M3FD"]
```

这里表示论文使用的数据集范围，不代表 `libs/data.py` 已经完整实现了所有数据集
的读取方式。使用新数据集前，请先检查数据加载器。

典型的 MSRS 目录结构如下：

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

具体目录名称以 [`libs/data.py`](libs/data.py) 的实现为准。

文本提示选项为 `discussion`；不使用文本提示时可以省略该参数，或者显式传入
`--prompt None`。启用文本提示时，需要将 `discussion.txt` 放在每个数据集划分目录中。

## 训练

论文采用 96x96 随机裁剪、batch size 为 2、训练 100 个 epoch，使用学习率为
`1e-4` 的 AdamW 优化器。PATS 使用 patch size 为 32、`K_pos=2`、`K_neg=8`，
温度参数为 `0.1`。

在仓库根目录执行：

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

当前 PATS 实现需要 batch size 为 2，因为每次训练需要构造同位和错位的
红外-可见光样本对。

## 测试

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

测试前需要在默认输出目录中准备训练权重，或者使用 `--weight_path` 指定权重路径。

## 指标计算

PTFuse 的训练和推理流程不再内置指标计算。如果需要计算评价指标，请参考：

- [RollingPlain/IVIF_ZOO](https://github.com/RollingPlain/IVIF_ZOO)
- [Zhaozixiang1228/MMIF-CDDFuse](https://github.com/Zhaozixiang1228/MMIF-CDDFuse)

进行结果比较时，应统一数据集划分、图像转换方式、归一化方式、指标实现、
评价参数以及官方权重协议。

## 论文结果

英文 README 中的结果表按论文原始顺序完整列出，包含：

- 三个数据集（MSRS、FMB、M3FD）上所有方法的 VIF、EN、SD、SF、AG、EI；
- M3FD 下游目标检测中各类别的 mAP@50、mAP@50:95 及平均值；
- FMB 和 MSRS 下游语义分割中各类别 IoU 及 mIoU；
- 同时展示 TeMAL（I-III）和 PATS（IV-VII）的完整消融表。

请查看英文 README 的 [完整实验结果](README.md#results)。表格数值均来自
`bare_jrnl_new_sample4.tex`；进行新的方法比较时，请按照上面的评价协议重新计算指标。

### 定性实验结果

![MSRS融合结果](figs/fused_MSRS.png)

*MSRS 低照度场景的融合结果对比。*

![FMB融合结果](figs/fused_FMB.png)

*FMB 跨数据集融合结果对比。*

![M3FD融合结果](figs/fused_M3FD.png)

*M3FD 雨天场景的融合结果对比。*

![M3FD检测结果](figs/detect_M3FD.png)

*M3FD 下游目标检测结果对比。*

![FMB分割结果](figs/seg_FMB.png)

*FMB 下游语义分割结果对比。*

![MSRS分割结果](figs/seg_MSRS.png)

*MSRS 下游语义分割结果对比。*

## 引用

如果使用本代码或 PTFuse，请引用 IOTJ 论文：

```bibtex
@article{liu2026ptfuse,
  title   = {PTFuse: Prompt-Guided Third-Party Modality Exploration for Infrared-Visible Image Fusion},
  author  = {Liu, Jianpu and Yang, Yang and Lang, Yue and Miao, Di and Chen, Mo},
  journal = {IEEE Internet of Things Journal},
  year    = {2026},
  url     = {https://ieeexplore.ieee.org/abstract/document/11598884}
}
```

如果有任何问题可以通过邮件联系我: liujianpu@tju.edu.cn.

## 致谢

感谢论文中使用的对比算法及其公开实现。原始论文出处如下，使用或比较这些方法时
请引用对应工作：

- **LRRNET** — TPAMI 2023。[论文](https://ieeexplore.ieee.org/document/10105495)
- **CDDFuse** — CVPR 2023。[论文](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_CDDFuse_Correlation-Driven_Dual-Branch_Feature_Decomposition_for_Multi-Modality_Image_Fusion_CVPR_2023_paper.html)
- **TGFuse** — TIP 2023。[论文](https://ieeexplore.ieee.org/document/10122870)
- **DDFM** — ICCV 2023。[论文](https://openaccess.thecvf.com/content/ICCV2023/html/Zhao_DDFM_Denoising_Diffusion_Model_for_Multi-Modality_Image_Fusion_ICCV_2023_paper.html)
- **DDBF** — CVPR 2024。[论文](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_Dispel_Darkness_for_Better_Fusion_A_Controllable_Visual_Enhancer_based_CVPR_2024_paper.html)
- **EMMA** — CVPR 2024。[论文](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_Equivariant_Multi-Modality_Image_Fusion_CVPR_2024_paper.html)
- **Text-IF** — CVPR 2024。[论文](https://openaccess.thecvf.com/content/CVPR2024/html/Yi_Text-IF_Leveraging_Semantic_Text_Guidance_for_Degradation-Aware_and_Interactive_Image_CVPR_2024_paper.html)
- **FreqGAN** — TCSVT 2025。[论文](https://ieeexplore.ieee.org/document/10680110)
- **TextFusion** — Information Fusion 2025。[论文](https://doi.org/10.1016/j.inffus.2025.103046)
- **PromptF** — JAS 2025。[论文](https://ieeexplore.ieee.org/document/10815008)
