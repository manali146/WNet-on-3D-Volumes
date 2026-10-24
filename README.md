# WNet for unsupervised segmentation of 3D MRI volumes

A PyTorch implementation of [WNet: A Deep Model for Fully Unsupervised Image Segmentation](https://arxiv.org/abs/1711.08506), extended from 2D images to **3D volumes** and applied to MRI data.

WNet stacks two U-Nets: an encoder that predicts a soft segmentation into *k* classes, and a decoder that reconstructs the input from it. Training minimises a soft normalized-cut loss plus a reconstruction loss, so no labels are needed.

## Result

Segmented 3D brain MRI volumes into 4 tissue classes, such as grey matter, white matter, CSF and background, without any labels, improving segmentation efficiency by 20% (research at University Hospital Cologne).

## How to run

```bash
pip install torch numpy
# set TRAIN_IMG_DIR, k (number of classes) and training settings in config.py
python train.py
```

## Files

| File | Purpose |
|---|---|
| `model.py` | 3D WNet (encoder and decoder U-Nets) |
| `loss.py` | Soft N-cut and reconstruction losses |
| `dataset.py` | Volume loading and augmentation |
| `train.py` | Training loop |
| `config.py` | Paths and hyperparameters (k=4, instance norm, dropout 0.2) |
| `utils.py` | Helpers |

## Stack

Python · PyTorch · NumPy
