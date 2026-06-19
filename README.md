# Faster R-CNN for Underwater Object Detection

A from-scratch PyTorch implementation of **Faster R-CNN** for detecting marine creatures in underwater imagery. Built as the final project for the *Machine Learning (Fall 2025)* course — every core component (backbone, RPN, ROI head, anchor generation, NMS, mAP, training loop, mosaic augmentation) is implemented from the ground up. **No pretrained weights and no `torchvision.models.detection` modules are used.**

---

## 🐟 Task

Given an RGB underwater image, predict for every visible creature:

- a **class label** (one of 7 marine species),
- a **bounding box** `(xmin, ymin, xmax, ymax)`, and
- a **confidence score**.

Detections are post-processed with a custom **Soft-NMS** implementation.

| Component       | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| **Input**       | RGB underwater image                                         |
| **Output**      | Set of `(class, bbox, confidence)` per image                 |
| **Model**       | Faster R-CNN — implemented from scratch                      |
| **Post-process**| Soft-NMS — implemented from scratch                          |

---

## 📦 Dataset

[**Aquarium / Underwater Object Detection**](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots) (Kaggle, via `kagglehub`).

- **638** underwater images across **7 classes**:
  `fish`, `jellyfish`, `penguin`, `shark`, `puffin`, `stingray`, `starfish`
- Provided `train` / `valid` / `test` splits are used as-is.
- Annotations are read and converted from scratch (no off-the-shelf parser).

The dataset is automatically downloaded inside the notebook:

```python
import kagglehub
path = kagglehub.dataset_download("slavkoprytula/aquarium-data-cots")
```

---

## 🏗️ Project Structure

```
faster-rcnn-underwater-detection/
└── UnderwaterObjectDetection.ipynb   # End-to-end notebook (all stages, all outputs preserved)
```
---

## 🚀 Quick Start

The notebook is designed to run on **Google Colab** with a GPU runtime.

1. Open `UnderwaterObjectDetection.ipynb` in Colab (or Jupyter).
2. Install Kaggle Hub if not already available:
   ```bash
   pip install kagglehub
   ```
3. Run all cells — the dataset is downloaded automatically, the model is built and trained, and qualitative + quantitative results are produced.

### Requirements

- Python ≥ 3.10
- PyTorch (CUDA recommended)
- `numpy`, `pandas`, `matplotlib`, `Pillow`, `opencv-python`, `tqdm`, `kagglehub`

---

## 📚 Notebook Outline

The notebook mirrors the assignment's 8 stages:

1. **Theory** — R-CNN ↔ Fast R-CNN ↔ Faster R-CNN comparison; one-stage vs two-stage detectors; deep dive on **GIoU**, **Soft-NMS**, **OHEM** and why they help underwater data.
2. **EDA & preprocessing** — sample visualizations, class distribution, objects-per-image, box-size distribution, preprocessing decisions.
3. **Data augmentation** — basic augmentations + custom **Mosaic** with before / after visualizations.
4. **Dataloader & custom `collate_fn`** — padding + mask demonstration.
5. **Faster R-CNN architecture** — custom backbone, RPN, ROI Pool / Align, detection head; tensor shapes traced end-to-end.
6. **RPN implementation** — anchor generation, positive / negative labeling, sampling.
7. **Training & loss** — multi-part loss, training loop, loss & mAP curves.
8. **Results** — quantitative (mAP) + qualitative (visualizations) + error analysis.

---


## 👤 Authors

**Mozhan Homavandi** 

**Maryam Mehdizade**

— Machine Learning, Fall 2025.

---
