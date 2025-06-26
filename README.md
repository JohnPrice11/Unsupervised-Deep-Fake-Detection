# Improved Unsupervised DeepFake Detection

This repository builds upon the original [@bestalllen/Unsupervised_DF_Detection](https://github.com/bestalllen/Unsupervised_DF_Detection) project, addressing its limitations and introducing several enhancements for more effective unsupervised DeepFake detection using deep learning.

---

## Table of Contents

- [Introduction](#introduction)
- [Original Limitations](#original-limitations)
- [Our Improvements](#our-improvements)
- [Datasets](#datasets)
- [Results](#results)
- [Changelog](#changelog)
- [Getting Started](#getting-started)
- [References](#references)

---

## Introduction

DeepFake videos pose significant challenges to online media integrity. This project aims to improve the state-of-the-art in unsupervised DeepFake detection by enhancing an existing approach, introducing missing stages, optimizing GPU utilization, improving feature visualization, and boosting classification accuracy through architectural changes.

---

## Original Limitations

The baseline repository had several limitations:
- Missing code for stage 3 (binary classifier training)
- Inefficient GPU utilization in stage 2
- Lack of t-SNE feature visualization
- No use of pretrained weights on the backbone model
- Required a large number of videos for testing
- Used Xception as the backbone

---

## Our Improvements

**Key enhancements and fixes:**

- **Stage 3 implementation:** Built the missing code for binary classifier training.
- **GPU optimization:** Improved GPU utilization for faster and more efficient training in stage 2.
- **t-SNE Visualization:** Added t-SNE plotting for visualizing feature separability and cluster assignments.
- **Checkpointing:** Implemented checkpoint saving in stage 2 for more robust training.
- **Clustering assignment:** Modified cluster assignment logic for better alignment with the original paper ("Assign Real and Fake to the Clusters").
- **Architectural changes:**
  - Swapped Xception backbone for `ConvNeXt_base` with pretrained weights.
  - Replaced Spearman correlation with Euclidean distance for inter-frame correlation.
  - Redesigned stage 3 as a binary classifier.
  - Added stage 4 for video authentication/testing.

---

## Datasets

Preprocessing includes frame extraction and labelling:

| Dataset      | Real (R) | Fake (F)       |
|--------------|----------|----------------|
| FF++         | 1000     | 4000 (DF, NT, F2F, FS) |
| UADFV        | 49       | 49             |
| Celeb-DF     | 400      | 800            |
| Celeb-DF-v2  | 1000     | 5600           |

---

## Results

### Comparative Results

| Training Set  | UADFV | CelebDF | CelebDF-v2 |         |
|---------------|-------|---------|------------|---------|
| FF++ (Baseline) | 78    |   --    | 70         | Baseline |
| FF++ (Ours)     | 70.41 | 78.80   | 85.88      | Ours    |
| CelebDF-v2 (Baseline) | -- | -- | --         | Baseline |
| CelebDF-v2 (Ours)     | 89.80 | -- | --       | Ours    |

### Distribution Plots

- **Left:** Original (Spearman correlation)
- **Right:** Our improvement (Euclidean distance, ConvNeXt_base)
  
![Screenshot from 2025-06-26 12-52-36](https://github.com/user-attachments/assets/5f8925ac-2541-44af-b3ef-e40117d251e8)

### t-SNE Feature Visualization

- Left: Epoch 0  |  Right: Epoch 16
  
![Screenshot from 2025-06-26 12-52-12](https://github.com/user-attachments/assets/4397dd75-7a89-4865-905f-8424b766bd5b)

---

## Changelog

- Built missing stage 3 for binary classification
- Added efficient GPU usage in stage 2
- Added t-SNE visualization and checkpointing
- Changed clustering assignment as per original paper
- Replaced Xception with ConvNeXt_base and pretrained weights
- Used Euclidean distance instead of Spearman correlation
- Implemented stage 4 for authentication/testing

---

## Getting Started

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Improved_Unsupervised_DF_Detection.git
    cd Improved_Unsupervised_DF_Detection
    ```

2. **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare datasets:**  
    Follow the dataset preprocessing instructions in the [Datasets](#datasets) section.

4. **Run training pipeline:**  
    Refer to the scripts for each stage. Example for stage 2:
    ```bash
    python stage2_train.py --config configs/stage2.yaml
    ```

5. **Visualizations and results:**  
    Check `outputs/` for plots and metrics.

---

## References

- [Original Repository](https://github.com/bestalllen/Unsupervised_DF_Detection)
- [ConvNeXt Paper](https://arxiv.org/abs/2201.03545)
- [FF++ Dataset](https://github.com/ondyari/FaceForensics)
- [Celeb-DF Dataset](https://github.com/yuezunli/celeb-deepfakeforensics)

---

> **Contact:** For questions or contributions, please open an issue or submit a pull request.
