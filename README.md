# Vision Transformer Implementation and Knowledge Distillation

This project implements a Vision Transformer (ViT) model for image classification, along with knowledge distillation for model compression.

## Features

- Vision Transformer implementation with PyTorch
- Knowledge distillation from a teacher model to a student model
- Model pruning support
- Training progress visualization with progress bars
- Checkpoint saving and loading
- Comprehensive logging system

## Installation

1. Create a conda environment:
```bash
conda create -n VIT python=3.8
conda activate VIT
```

2. Install dependencies:
```bash
pip install torch torchvision tqdm
```

## Project Structure

```
.
├── configs/
│   ├── base_config.py
│   ├── model_config.py
│   └── train_config.py
├── scripts/
│   └── run.py
├── src/
│   ├── models/
│   │   └── nets/
│   │       ├── vit.py
│   │       └── vit_pruned.py
│   ├── trainers/
│   │   └── standard.py
│   └── utils/
│       ├── logger.py
│       └── utils_fit.py
└── outputs/
    ├── checkpoints/
    └── logs/
```

## Usage

### Training the Base Model

To train the baseline Vision Transformer model:

```bash
python scripts/run.py --mode train --model baseline --epochs 200
```

### Evaluating a Model

To evaluate a trained model:

```bash
python scripts/run.py --mode evaluate --model baseline --checkpoint outputs/checkpoints/checkpoint_baseline.pth
```

### Knowledge Distillation

To perform knowledge distillation from a trained teacher model to a student model:

```bash
python scripts/run.py --mode distill --checkpoint outputs/checkpoints/checkpoint_baseline.pth --epochs 200
```

## Model Architecture

### Vision Transformer (ViT)
- Patch size: 16x16
- Hidden dimension: 768
- Number of transformer layers: 12
- Number of attention heads: 12
- MLP size: 3072

### Pruned Model
- Reduced model size with similar performance
- Knowledge distillation from the base model

## Training Details

- Optimizer: SGD with momentum
- Learning rate scheduling: Cosine annealing
- Data augmentation: Random crop, horizontal flip
- Batch size: 64
- Training epochs: 200

## Logging and Checkpoints

- Training logs are saved in `outputs/logs/`
- Model checkpoints are saved in `outputs/checkpoints/`
- Both best and latest models are saved during training
- Progress bars show real-time training metrics

## Contributing

Feel free to submit issues and enhancement requests!
