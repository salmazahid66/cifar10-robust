# Robust CIFAR-10 Classification with Only 500 Training Images

A small Inception-style CNN (73,306 trainable parameters) trained from scratch on 400 CIFAR-10 images (40 per class) and validated on 100 (10 per class). It is tested on a shifted test set (rotation, brightness/contrast, blur, noise, ~25% occlusion and a hidden colour corruption).

## Results (mean ± std over 3 seeds)

| Configuration | Shifted-Val | Clean Test | Shifted Test |
|---|---|---|---|
| No augmentation (baseline) | 15.37 ± 0.45 | 30.95 ± 3.55 | 13.68 ± 0.45 |
| Geometric only | 17.67 ± 0.12 | 40.77 ± 1.38 | 16.13 ± 0.54 |
| Photometric only | 20.10 ± 0.45 | 40.10 ± 3.60 | 16.78 ± 0.97 |
| Geometric + photometric | 22.77 ± 1.22 | 39.09 ± 1.52 | 18.69 ± 1.07 |
| + Cutout / Random Erasing (**final model**) | 25.50 ± 1.77 | 39.46 ± 1.25 | **24.10 ± 0.71** |
| + Mixup + label smoothing | 23.47 ± 0.74 | 37.59 ± 0.25 | 22.81 ± 0.41 |

- **Final model:** geometric + photometric + Cutout, shifted test 24.10 ± 0.71%.
- **Largest single gain:** Cutout / Random Erasing (+5.4 points), which matches the ~25% rectangular occlusion in the shifted test.
- Hidden-corruption-only test (stand-in corruption): final 31.81 ± 1.46% vs baseline 27.42 ± 2.36%.
- Test-time augmentation (6 views) did not help: 24.10% → 24.05%, with about 4.5x inference time.

## Model

- Stem (two 3x3 convs) → Inception module (1x1, 3x3, 5x5 branches with 1x1 reductions + pooling branch) → Inception module → Global Average Pooling → Dropout (0.3) → Linear(160, 10)
- BatchNorm after every conv. 73,306 trainable parameters (limit: 1,000,000). No pretrained weights.

## Training setup

- AdamW, lr = 0.003, weight decay = 0.005, OneCycle schedule, batch size 32
- Maximum 60 epochs, early stopping (patience 25) on shifted-validation accuracy
- Augmentation is applied on the fly to training data only (no pre-generated copies)
- Hyperparameters (lr, weight decay) were chosen on the validation split only; test data was never used for tuning or model selection

## Seeds and splits

| Item | Value |
|---|---|
| Train/validation split seed | 42 (stratified: 50 images per class, 40 train + 10 validation) |
| Shifted test corruption seed | 1234 |
| Hidden-corruption-only test seed | 1235 |
| Shifted validation (10 corrupted copies of the 100 val images) | seeds 1236 to 1245 |
| Training seeds | 0, 1, 2 |

## How to reproduce

1. Open `CIFAR10_robust_classification.ipynb` in Google Colab (Runtime → Change runtime type → GPU) or in a local Jupyter environment.
2. Install the requirements if needed:
```
   pip install torch torchvision numpy pandas matplotlib scikit-learn pillow
```
3. Run all cells from top to bottom. CIFAR-10 is downloaded automatically through `torchvision`. The full ablation (6 configurations x 3 seeds) takes a while, so a GPU is recommended.
4. Do not re-run the setup/utilities cells between the sweep and the ablation, so the selected hyperparameters stay unchanged.

## Notes and limitations

- **The hidden corruption is a stand-in** (random grayscale or random channel swap), implemented in `hidden_corruption()`, because the official one was not available. Replace that function with the official implementation and re-run to reproduce results on the real hidden corruption.
- The shifted-test pipeline is my own implementation of the described corruptions (rotation up to ±25°, brightness/contrast, blur, noise, ~25% black occlusion), so absolute numbers may differ from other implementations.
- The validation split has only 100 images, so small differences (a few points) between configurations are within noise.
- The "Shifted-Val" column above is accuracy on 10 corrupted copies of the 100 validation images, and it is the score used for model selection.