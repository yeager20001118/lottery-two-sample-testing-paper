# LoTT: Learning from Reference-Only Samples in Two-Sample Testing under Size Asymmetry

Official implementation of **LOTTERY (LoTT)**, accepted at **ICML 2026**.

> **LOTTERY: Learning from Reference-Only Samples in Two-Sample Testing under Size Asymmetry**
> Xunye Tian*, Zhijian Zhou*, Liuhua Peng, Feng Liu
> *Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026.*

## Overview

LoTT addresses the fundamental challenge of two-sample testing when reference data is abundant but query data is extremely scarce (the *size-asymmetry* regime). Unlike existing methods that split both samples for training and testing, LoTT learns entirely from reference data, reserving all query samples for testing.

**Key idea**: Instead of learning a discrepancy between X and Y, LoTT learns a collection of *Reference-Dependent Representations* (RDRs) from X alone, then tests whether Y is compatible with these learned representations.

### Method Summary

1. **Reference-Only Learning** (Section 4): Partition reference sample X into X^tr (training), X^cal (calibration), and X^hold (holdout). Learn multiple RDR families from X^tr that capture different aspects of the reference distribution:

  | RDR Family          | What it captures          | Definition                                    |
  | ------------------- | ------------------------- | --------------------------------------------- |
  | **ME-RDR**          | Local similarity patterns | Kernel mean embedding distance (consistent)   |
  | **Mahalanobis-RDR** | Global structure          | Mahalanobis distance from reference center    |
  | **kNN-RDR**         | Local scale geometry      | Mean k-nearest-neighbor distance              |
  | **LOF-RDR**         | Local relative density    | Local Outlier Factor score                    |
  | **Landmark-RDR**    | Localized departures      | Kernel similarity to random reference subsets |

2. **Uncertainty-Guided Aggregation** (Section 5): Estimate each RDR's stability via resampling on X^cal. Weight RDRs by precision (inverse variance) to downweight unstable ones, tightening permutation thresholds.
3. **Pooled Permutation Testing** (Section 4.4): Pool X^hold and query Y, then permute to obtain a valid p-value. This guarantees exact finite-sample Type I error control (Theorem 6.1) and consistency (Theorem 6.2).

## Repository Structure

```
clean_code/
|-- lott/                       # Core LoTT implementation
|   |-- rdr.py                  # RDR families (Section 4.2)
|   |-- lott.py                 # LoTT and LoTTWithSelection (Sections 4-5)
|   |-- wrapper.py              # Experiment runner
|   +-- __init__.py
|-- models/                     # Embedding extractors for CIFAR-10
|   |-- resnet.py               # ResNet-18
|   |-- wide_resnet.py          # WideResNet-28x10
|   |-- __init__.py
|   +-- checkpoint/             # Place pretrained weights here
|-- exp/                        # Reproducible experiments
|   |-- dataloader.py           # Dataset loading utilities
|   |-- varying_m/              # Table 2: fixed N, varying query size M
|   |   |-- blob.py
|   |   +-- cifar10_resnet.py
|   +-- varying_n/              # Table 1: fixed M, varying reference size N
|       |-- blob.py
|       +-- cifar10_resnet.py
|-- data/                       # Datasets (see data/README.md)
|   +-- README.md
|-- requirements.txt
+-- README.md
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Minimal Example

```python
import torch
from lott import LoTTWithSelection

# Generate toy data: reference X ~ P, query Y ~ Q
X = torch.randn(500, 10)              # abundant reference
Y = torch.randn(20, 10) + 0.3         # scarce query (shifted)

# Split reference into train / calibration / holdout
n = len(X)
perm = torch.randperm(n)
X_train = X[perm[:200]]
X_calib = X[perm[200:250]]
X_hold  = X[perm[250:]]

# Fit LoTT (learns RDRs from reference only)
lott = LoTTWithSelection(alpha=0.05, n_permutations=500,
                          selection_method='precision_weight', verbose=False)
lott.fit(X_train, X_calib, X_hold)

# Test query batch
result = lott.test(Y)
print(f"Reject H0: {result['reject']}, p-value: {result['p_value']:.4f}")
```

## Reproducing Paper Results

### BLOB (Synthetic)

```bash
# Table 2 (varying M): test power and type I error
cd exp/varying_m
python blob.py --check=1    # test power
python blob.py --check=0    # type I error

# Table 1 (varying N): test power
cd exp/varying_n
python blob.py --check=1
```

### CIFAR-10 (Adversarial Detection)

Requires pretrained models and adversarial data (see `data/README.md`).

```bash
# Table 2 (varying M)
cd exp/varying_m
python cifar10_resnet.py --check=1 --model_arch=Res18

# Table 1 (varying N)
cd exp/varying_n
python cifar10_resnet.py --check=1 --model_arch=Res18
```

### SLURM Submission

```bash
cd exp/varying_m && sbatch run.slurm
cd exp/varying_n && sbatch run.slurm
```

## Baselines

We compare against the following methods. Their implementations can be found at:


| Method       | Type     | Reference                                 | Code                                                                                                                     |
| ------------ | -------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **MMD-M**    | Kernel   | Garreau et al., 2017                      | [AdapTesting](https://github.com/yeager20001118/AdapTesting)                                                             |
| **MMD-FUSE** | Kernel   | Biggs et al., 2023                        | [mmdfuse](https://github.com/antoninschrab/mmdfuse)                                                                      |
| **MMDAgg**   | Kernel   | Schrab et al., 2023                       | [mmdagg](https://github.com/antoninschrab/mmdagg)                                                                        |
| **C2ST-L**   | Learning | Lopez-Paz & Oquab, 2017; Kim et al., 2021 | [AdapTesting](https://github.com/yeager20001118/AdapTesting)                                                             |
| **MMD-Deep** | Learning | Liu et al., 2020                          | [DK-for-TST](https://github.com/fengliu90/DK-for-TST)                                                                    |
| **RL-TST**   | Learning | Tian et al., 2025                         | [RL-TST](https://github.com/yeager20001118/A-Unified-Data-Representation-Learning-for-Non-parametric-Two-Sample-Testing) |


## Citation

```bibtex
@inproceedings{tian2026lottery,
  title={LOTTERY: Learning from Reference-Only Samples in Two-Sample Testing under Size Asymmetry},
  author={Tian, Xunye and Zhou, Zhijian and Peng, Liuhua and Liu, Feng},
  booktitle={Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

This project is released for academic research purposes.