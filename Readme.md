# GSoC Task 2h: Linear Attention Vision Transformers for Mass Regression and Classification

This repository contains the implementation for **Specific Task 2h** for the CERN ML4Sci Google Summer of Code application. The goal of this task is to train a linear-scale attention Vision Transformer (ViT) on High-Energy Physics (HEP) jet data, finetune it for simultaneous classification and mass regression, and compare the results against a model trained entirely from scratch.

## Project Overview
- **Architecture:** Custom Vision Transformer using a Linear Attention mechanism ($O(N \cdot D^2)$ complexity) via the $\phi(x) = ELU(x) + 1$ feature mapping, ensuring high-speed processing of long sequences without standard self-attention memory bottlenecks.
- **Pretraining:** Self-supervised Masked Autoencoder (MAE) style pretraining on 60,000 unlabelled samples.
- **Finetuning vs. Scratch:** The pretrained model is finetuned on an 8,000-sample labelled dataset using a low learning rate. It is then benchmarked against an identical ViT initialized with random weights and trained from scratch.
- **Optimization:** Includes global normalization, mass scaling (to balance BCE and MSE losses), gradient clipping, and pure FP32 precision to prevent loss overflows.

## Repository Structure

```text
.
├── Task_2h_Linear_Transformer.ipynb   # Main Jupyter Notebook with full training pipeline
├── finetuned_linear_vit.pth           # Saved weights of the finetuned model
├── scratch_linear_vit.pth             # Saved weights of the scratch model
├── checkpoints/                       # (Optional) Intermediate epoch weights
└── results/
    └── comparison_graph.png           # Pre-saved validation loss comparison plot
```

## Important Note on Results & Plotting

Both the **finetuned** and **scratch** model weights have been successfully saved (`.pth` files) and are included in this repository.

Generating the final `matplotlib` comparison graph requires continuous epoch-by-epoch validation loss tracking, which is maintained in memory during the training process. To avoid the need to re-run the entire training pipeline solely for visualization, the final convergence comparison graph has been pre-generated and is available in the `results/` directory.

For full reproducibility, the notebook may be executed sequentially from start to finish.
## How to Run
1. Ensure you have the required dependencies installed:
   ```bash
   pip install torch torchvision h5py numpy matplotlib tqdm
