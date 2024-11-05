# ml-bench-rct

Exploring ML model comparison through randomized experimentation.

## Overview

This project for [Prof. Don Green](https://donaldgreen.com/)'s randomized experimentation class at Columbia (POLS 4724) investigates an alternative approach to comparing machine learning models under computational constraints. When evaluating ML algorithms, researchers often face a fixed computational budget that prevents running all models on all available benchmark datasets. The conventional approach is to select a subset of benchmarks and evaluate all models on that subset. However, this may fail to capture performance differences that only emerge on the unused datasets.

This project explores an unconventional alternative: using randomized controlled trials (RCTs) to allocate datasets to models. Given a fixed evaluation budget and a desire to understand model performance across a variety of contexts, we investigate whether random assignment might better recover true performance differences compared to the traditional approach of running all models on a smaller, selected subset of benchmarks.

## Methodology

This project investigates model comparison through the lens of experimental design:

1. Select 30 image classification datasets from torchvision as our benchmark population
2. Given a computational budget insufficient for running both models on all datasets, randomly assign each dataset to either:
   - Control: Convolutional Neural Network (CNN)
   - Treatment: Modern Vision Transformer (ViT)
3. Use blocked randomization based on dataset characteristics to ensure balanced comparison
4. Analyze results using methods from experimental research

The central research question explores whether this randomized approach could better recover true performance differences between models compared to evaluating both models on a smaller subset of benchmarks. While admittedly unconventional (since we could theoretically run both models on all datasets), this investigation aims to contribute to the broader discussion of systematic approaches to model evaluation under real-world computational constraints.

## Project Organization

```
.
├── analysis/    # R code for randomization and experimental analysis
├── doc/         # Documentation and pre-analysis plan
├── src/         # Python implementation of dataset loading and model training
└── tools/       # One-off utility scripts for dataset preparation
```

## Further Reading

For more details about the experimental design and analysis plan, see [doc/pre-analysis.pdf](doc/pre-analysis.pdf).
