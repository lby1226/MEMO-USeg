<h1 align="center">MEMO-USeg</h1>

<p align="center">
  <b>An Organ-Conditioned Self-Prompting Multi-Expert Framework for Multi-Organ Ultrasound Image Segmentation</b>
</p>

<p align="center">
  <img alt="Status" src="https://img.shields.io/badge/Status-Under%20Review-orange">
</p>

---

MEMO-USeg is an organ-conditioned, self-prompting multi-expert framework for multi-organ ultrasound image segmentation. It enables a single model to jointly learn multiple organ segmentation tasks and can be extended to new organs after training by adding organ-specific experts (SOEM).

## 📄 Paper

This repository provides the official implementation of MEMO-USeg.

**MEMO-USeg: An Organ-Conditioned Self-Prompting Multi-Expert Framework for Multi-Organ Ultrasound Image Segmentation** — **Under Review**

### ✨ Method Highlights

- **One model, multiple organs.** Through multi-expert joint learning on multi-organ images, a single model performs automatic segmentation across multiple organs.
- **Broad organ coverage.** Validated on ultrasound images covering 13 organs.
- **Extensible organ-incremental learning.** By adding organ-specific experts (SOEM) and freezing previously trained SOEMs, the model preserves its ability on existing organs while adapting to new ones.
- **Prompt-free deployment.** No manual bounding box or point prompts are required.

<p align="center">
  <img src="assets/organs.png" alt="MEMO-USeg covers 13 ultrasound organs" width="300">
</p>

<p align="center"><i>Examples from the 13 ultrasound organs covered by MEMO-USeg.</i></p>

## 🏥 Clinical Application

In clinical use, the operator only needs to select the target organ in the MEMO-USeg interface and upload the corresponding ultrasound image; the organ is then segmented automatically. A single model handles multi-organ segmentation without any manual box or point prompts.

<p align="center">
  <img src="assets/demo.gif" alt="MEMO-USeg clinical application demo" width="85%">
</p>

<p align="center"><i>Select an organ and upload an ultrasound image — segmentation is fully automatic.</i></p>

## ⚙️ Installation and Getting Started

> [!IMPORTANT]
> **Details of the model architecture and its associated weights are being curated and will be released following the acceptance of this manuscript.**

### 1. Environment

```bash
git clone https://github.com/lby1226/MEMO-USeg.git
cd MEMO-USeg
conda create -n memo-useg python=3.10 -y
conda activate memo-useg
pip install -r code/requirements.txt
```

If you need a specific CUDA build of PyTorch, install it first from the [PyTorch index](https://pytorch.org/get-started/locally/), for example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Dataset preparation

Arrange each ultrasound dataset under a common root, with images and masks sharing the same file stem:

```text
${DATASET_ROOT}/
├── TN3K/
│   ├── trainval/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
├── BUSI/
│   └── ...
└── HMC-QU/
    └── ...
```

The 13 datasets and their class ids are defined in [`code/datasets.yaml`](code/datasets.yaml). Point `DATASET_ROOT` to your data directory:

```bash
export DATASET_ROOT=/path/to/datasets     # Windows: set DATASET_ROOT=D:\datasets
```

### 3. Pretrained backbone

Download the official [SAM 2.1](https://github.com/facebookresearch/sam2#download-checkpoints) Hiera checkpoint (e.g. `sam2.1_hiera_base_plus.pt`, passed via `--sam2_ckpt`) or [SAM 1](https://github.com/facebookresearch/segment-anything#model-checkpoints) ViT checkpoint (`sam_vit_*.pth`, passed via `--sam_ckpt`) into `./checkpoints`.

### 4. Training

```bash
python code/train_mome_useg.py \
    --arch sam2 --model_type base \
    --data_config code/datasets.yaml \
    --gpu 0 --epochs 50 --batch_size 32 \
    --save_dir work_dir/MEMO_SAM2_5Fold
```

The script runs 5-fold cross-validation and keeps the best checkpoint according to validation Dice. Default settings: `--common_ratio 4.0` (R_GE), `--expert_ratio 0.25` (R_SE) and `--prompt_threshold 0.9` (τ).

### 5. Evaluation

```bash
python code/test_mome_useg.py \
    --arch sam2 --model_type base \
    --checkpoint_dir work_dir/MEMO_SAM2_5Fold \
    --gpu 0
```

Dice, IoU, HD95 and ASSD are computed for every dataset and reported as mean ± std over the evaluated folds, together with the macro average across datasets.
