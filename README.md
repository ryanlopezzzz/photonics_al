# Active Learning for Photonic Crystals

This repository contains code for **Active learning for photonic crystals**, using analytic approximate Bayesian last-layer neural networks (LL-BNNs) to guide uncertainty-driven sample acquisition.

The goal is to reduce the number of expensive band-structure simulations required to accurately predict photonic band gaps by prioritizing the most informative structures during training.

The dataset was generated the same way as in "Surrogate- and invariance-boosted contrastive learning for data-scarce applications in science" [Github](https://github.com/clott3/SIB-CL) | [Paper](https://doi.org/10.1038/s41467-022-31915-y)

## Repository Structure

```text
photonics_al/
├── dataset/                 # Code for loading and plotting PhC dataset
├── images/                  # Plots generated from plot_results.ipynb
├── active_learning.py       # Runs active learning loop
├── augmentations.py         # Symmetry operations that preserve band gap
├── bayesian_layer.py        # Analytic approximate final linear Bayesian layer
├── model_architecture.py    # Neural network architecture
├── plot_results.ipynb       # Plotting for figures in paper
├── submit_cluster_jobs.py   # Specific for MIT SuperCloud cluster
└── train_model.py           # Train individual model for band gap prediction