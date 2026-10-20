# GraphSG

Code for **From Graphs to Hypergraphs: A Shared Representation-based Framework for Size Generalization (GraphSG)**.

## Introduction

We propose GraphSG, a unified framework for graph size generalization through task-invariant shared representation learning. First, we construct multi-size graph views to capture structural variations across different graph scales. Second, we align multi-view representations and extract task-relevant shared information to mitigate size-induced distribution shifts. Finally, we extend GraphSG to hypergraph classification through a task-aware hypergraph sampling mechanism that preserves task-relevant higher-order structural information.

## Dependencies

The implementation uses **Python 3.8** and the following packages:

```text
numpy==1.24.3
pandas==2.0.3
scipy==1.10.1
scikit-learn==1.3.2
networkx==3.1
ogb==1.3.6
torch==2.0.0
torchvision==0.15.0
torch-geometric==2.6.1
torch-scatter==2.1.2+pt20cu118
torch-sparse==0.6.18+pt20cu118
torch-cluster==1.6.3+pt20cu118
torch-spline-conv==1.2.2+pt20cu118
pyg-lib==0.4.0+pt20cu118
tqdm==4.67.2
```

Experiments were conducted on an **NVIDIA GeForce RTX 3090 GPU** using PyTorch 2.0.0 and CUDA 11.8-compatible PyTorch Geometric extensions.

## Usage

### Create an environment using Conda

```bash
conda create -n graphsg python=3.8
conda activate graphsg
```

Install the required dependencies using versions compatible with PyTorch 2.0.0 and CUDA 11.8.

### Run GraphSG

The repository contains scripts for data preprocessing, model training, and evaluation.

- `datapreprocessing.py`: Dataset preprocessing.
- `model.py`: Model implementation.
- `criteria.py`: Training objectives.
- `train.py`: Model training.
- `test.py`: Model evaluation.
- `main.py`: Main experiment entry point.

Please configure the dataset paths and experimental settings before running the corresponding scripts.

## Reproducibility

GraphSG is evaluated on graph and hypergraph classification benchmarks under graph size generalization settings. Dataset splits, hyperparameters, and evaluation protocols follow the experimental configurations described in the paper.

## Citation

If you find this work useful, please consider citing:

**From Graphs to Hypergraphs: A Shared Representation-based Framework for Size Generalization**

Citation information will be updated after publication.
