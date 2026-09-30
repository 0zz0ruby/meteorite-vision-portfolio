# Experiment log and evidence notes

This document separates measured results from reconstructed observations. It is
intended to make the portfolio repository honest and auditable even though the
original console logs were not preserved.

## Classification

### ResNet-18 baseline

- Built the first complete training, validation, threshold-search and Kaggle
  submission workflow.
- Local validation F1: **0.9852**.
- Stage-1 online F1: **0.72222**.
- The large local/online gap motivated investigation of background shortcuts
  and domain shift.

### DINOv2 optimization

- Used a frozen DINOv2 ViT-S/14 backbone and trained a linear classifier.
- Concatenated the CLS tokens from the final two transformer blocks, producing
  a 768-dimensional feature vector.
- Used image-hash groups in the validation split to reduce leakage from near
  duplicates.
- Best recovered checkpoint metadata: epoch 28, validation macro F1
  **0.920356**, batch size 128, learning rate 0.003 and weight decay 0.01.
- Best individual online F1: **0.73**.

The DINOv2 checkpoint metadata, size and SHA-256 are preserved; the 84.22 MiB
binary itself is deliberately kept outside the source repository.

### Ablations and data expansion

The recovered project report records these online results:

| Experiment | Online F1 | Interpretation |
|---|---:|---|
| Two final CLS tokens, ViT-S/14 | 0.73 | Best individual model |
| One CLS token | 0.70 | Less expressive frozen representation |
| ViT-B/14 | 0.68 | Increased capacity did not improve transfer |
| 100+ external images | 0.71 | External data introduced domain mismatch |
| Remove GaussianBlur | 0.70 | Blur augmentation improved robustness |

Progressive-resolution training was also tried and produced no meaningful
gain. Because per-run logs are missing, these values are reported as recovered
results rather than newly reproduced measurements.

### Ensemble

When repeated single-model tuning plateaued, the author proposed combining
models by voting to exploit complementary errors. The final four-model voting
pipeline was implemented collaboratively by the team.

A preserved screenshot documents this v2 configuration:

| Model | Low | High | Weight |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 0.4891 | 0.6124 | 0.2696 |
| Swin V2-S | 0.5862 | 0.6973 | 0.0981 |
| DINOv2 ViT-S/14 | 0.2460 | 0.8884 | 0.2489 |
| EfficientNetV2-S | 0.0905 | 0.4400 | 0.3834 |

The overall decision threshold was **-0.1750**, and the four weights sum to
1.0. This exact v2 submission achieved **0.78260** public F1. The course
leaderboard separately records the team's final result as **0.78284**, but the
fitting script and exact final parameter set were not retained. The repository
does not assume that the v2 numbers are necessarily the parameters behind the
0.78284 result.

The voting proposal is an individual contribution; parameter tuning, final
implementation and submission are team results.

## Meteorite image generation

The individual generation assignment used 2,682 foreground-isolated real
images and evaluated 1,000 generated samples with FID.

| Version | Main change | FID | Outcome |
|---|---|---:|---|
| DCGAN baseline | Initial adversarial baseline | 245.9168 | Starting point |
| WGAN-GP v1 | Gradient penalty | 189.7166 | More stable training |
| WGAN-GP v2 | Capacity rebalance | 170.82 | Better generator/discriminator balance |
| WGAN-GP v3 | Spectral normalization and tuning | 137.25 | Best reported comparison run |
| WGAN-GP v3 official | Final saved evaluation | **134.203409** | Best preserved metric |
| WGAN-GP v4 | EMA, larger model, stronger augmentation | 400+ | Regression; rejected |
| WGAN-GP v5 | Feature matching | — | Exploratory; no reliable final result |

The failed v4 and incomplete v5 notebooks are intentionally retained to show
experimental judgment rather than only a polished final score.

## Limitations

- Original epoch-by-epoch logs are unavailable, so tables distinguish preserved
  metrics from qualitative conclusions.
- The ensemble fitting script is unavailable. The v2 rule has been reconstructed
  from its preserved formula and parameter screenshot.
- Datasets and scraped images are excluded because of size, competition rules
  and uncertain redistribution rights.
- The classification competition was team-based; this repository labels the
  author's contribution separately from team outcomes.
