# GEGNN
A Graph Embedding-Enhanced GNN for Power Control in Device-to-Device Communications

This repository contains a PyTorch implementation of the architectures described in **"Graph Neural Networks for Scalable Radio Resource Management: Architecture Design and Theoretical Analysis"** ([arXiv:2007.07632](https://arxiv.org/abs/2007.07632)). 

Our code builds upon their experimental environment and proposed Message Passing Graph Neural Network (MPGNN) framework to optimize power control.

📁 Project Structure
```text
├── baselines/          # Non-ML optimization baselines and reference algorithms
├── config/             # Centralized hyperparameter and experiment configurations
├── experiments/        # Execution scripts for benchmarks and comparisons
├── graph/              # Graph construction and dataset
├── metrics/            # loss functions and evaluation
├── models/             # GNN models
├── simulation/         # Environment generation
└── training/           # Core training loops and validation
