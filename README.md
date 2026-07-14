# Chest X-Ray Pneumonia Classifier

A convolutional neural network that classifies chest X-rays as **normal** or **pneumonia**, built with PyTorch using transfer learning on a pretrained ResNet18.

The focus of this project is not just training a classifier, but reading its behaviour honestly — diagnosing where and why it fails, and correcting a class-imbalance bias that a headline accuracy number would have hidden.

## Overview

- **Task:** binary image classification (normal vs. pneumonia) from chest X-rays
- **Approach:** transfer learning — fine-tune an ImageNet-pretrained ResNet18 with a new 2-class head
- **Framework:** PyTorch, trained on GPU
- **Result:** 85% test accuracy, with near-zero false negatives on pneumonia (recall 0.99)

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — ~5,800 labeled pediatric chest X-rays, originally from Kermany et al. (2018). The data is split into `train`, `val`, and `test` folders, each with `NORMAL` and `PNEUMONIA` subfolders.

The dataset is **not** included in this repo (it's ~1–2 GB). Download it from the link above; the notebook expects it at `/kaggle/input/chest-xray-pneumonia/chest_xray/` (the default when run on Kaggle).

## Method

1. **Data pipeline** — `ImageFolder` loading, resize to 224×224, ImageNet normalization, and light augmentation on the training set.
2. **Model** — ResNet18 pretrained on ImageNet, with the final 1000-class layer replaced by a fresh 2-class linear layer (transfer learning).
3. **Training** — cross-entropy loss, Adam optimizer, fine-tuned over several epochs on GPU.
4. **Evaluation** — accuracy alongside per-class precision/recall and a confusion matrix, since accuracy alone is misleading on imbalanced medical data.

## Results

The training set is imbalanced (~3,900 pneumonia vs. ~1,340 normal), and this showed up directly in the model's behaviour: it was biased toward predicting pneumonia. Adding a **class-weighted loss** to counter the imbalance improved detection of healthy cases while keeping pneumonia detection essentially perfect.

Final test set (624 images):

| Class     | Precision | Recall | F1   |
|-----------|-----------|--------|------|
| Normal    | 0.98      | 0.62   | 0.76 |
| Pneumonia | 0.81      | 0.99   | 0.89 |
| **Accuracy** |        |        | **0.85** |

Of 390 pneumonia cases, only **3 were missed** (false negatives). The model's remaining errors are almost all false positives — healthy patients flagged for a second look — which is the safer direction to err in a screening context, where a missed diagnosis is far more costly than a false alarm.

## Key takeaway

The most useful part of the project was not the accuracy number but the diagnosis behind it: spotting the imbalance-driven bias in the confusion matrix, connecting it to the training data, and correcting it deliberately — then reasoning about the remaining errors in clinical terms (false negatives vs. false positives) rather than treating all mistakes as equal.

## Tech stack

Python · PyTorch · torchvision · scikit-learn · matplotlib

## Running it

The simplest path is on [Kaggle](https://www.kaggle.com/) with a GPU enabled and the dataset attached — open the notebook, enable a GPU accelerator, and run top to bottom. To run locally, download the dataset, update the data path near the top of the notebook, and ensure PyTorch is installed with CUDA if you want GPU training.
