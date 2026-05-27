# Data Preparation

## BLOB (Synthetic)

No external data needed. The BLOB dataset is generated on the fly via `sample_blob()` in `exp/dataloader.py`.

## CIFAR-10 (Adversarial Detection)

1. **Clean images**: Automatically downloaded by `torchvision.datasets.CIFAR10`.

2. **Adversarial images**: PGD adversarial examples (epsilon=4/255, L-inf) generated against the pretrained models.
   Place the following files in `data/cifar10/`:
   - `Adv_cifar10_pgd_5_eps4_linf.npz` (for ResNet-18)
   - `Adv_cifar10_pgd_5_eps4_linf_transfer_wrn28.npz` (for WRN-28)

   Each `.npz` file should contain:
   - `X_adv`: Adversarial images (float32, shape [N, 3, 32, 32])
   - `predicted_original_labels`: Original model predictions
   - `predicted_adv_labels`: Model predictions on adversarial images

3. **Pretrained model checkpoints**: Place in `models/checkpoint/`:
   - `resnet-18.pth` (ResNet-18 trained on CIFAR-10)
   - `wide-resnet-28x10.pth` (WideResNet-28x10 trained on CIFAR-10)

## Higgs

Download the preprocessed Higgs dataset and place as `data/HIGGS_TST.pckl`.
The pickle file contains a list `[X_signal, X_background]` of numpy arrays with 4-dimensional features.
