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

### ✨ Key Features

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

To be added.
