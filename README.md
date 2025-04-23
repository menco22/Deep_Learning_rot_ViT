# Self-Supervised Vision Transformer (ViT) with Rotation Prediction

## Project Overview

This project implements a self-supervised learning approach for Vision Transformers (ViT) using rotation prediction as a pretext task. The main goal is to train a ViT model without the need for large amounts of labeled data, leveraging the rotation prediction task to learn useful visual representations that can be transferred to downstream classification tasks.

## Architecture

The project uses a Vision Transformer (ViT) architecture with the following characteristics:
- **Image size**: 160x160 pixels
- **Patch size**: 16x16 pixels
- **Embedding dimension**: 512
- **Number of layers**: 8
- **Number of attention heads**: 8

## Methodology

The learning approach is divided into several phases:

### 1. Pre-training with Rotation Prediction

During the pre-training phase, the model is trained on ImageNet using a rotation prediction task. Each image is randomly rotated (0°, 90°, 180°, or 270°), and the model must predict the rotation angle. This forces the model to learn semantically meaningful features without requiring manual labels.

Pre-training parameters:
- **Optimizer**: AdamW (β₁=0.9, β₂=0.95)
- **Base learning rate**: 1.5e-4
- **Weight decay**: 0.05
- **Scheduling**: Warmup followed by cosine decay
- **Epochs**: 1 (there were supposed to be 5 but after the first one it crashed)

### 2. Transfer Learning on Imagenette

After pre-training, the model is fine-tuned on the Imagenette dataset (a smaller version of ImageNet with 10 classes) through various strategies:

#### a. Linear Probing
- Freezing all pre-trained weights
- Replacing and training only the final classifier

#### b. Frist Fine-Tuning 
Unfreezing the classifier and layer 7


### c. Second Fine-Tuning
Unfreezing the classifier, layer 7 and layer 6

### 3. Feature Analysis

The project also includes an analysis of the features learned at different levels of the transformer, evaluating how classification performance varies when using representations from different layers.

## Dataset

- **Pre-training**: ImageNet-1k
- **Fine-tuning and evaluation**: Imagenette (320px version)

## Implementation

The project is organized into the following main modules:

- **main.py**: Main script that runs the entire workflow
- **NOT_pre_trained_Vit.py**: Trains a ViT model from scratch (without pre-training)
- **featuresAnalysis.py**: Analyzes the features extracted from different layers
- **single_pre_train_experiments/**: Individual scripts for each experimental phase
  - **1_pre_Train_ROT_ViT.py**: Pre-training with rotation prediction
  - **2a_LP_RotVit_on_Imagenette.py**: Linear probing
  - **2b_FT_CH_L7_Rot_Vit_on_imagenette.py**: Fine-tuning the classifier and layer 7
  - **2c_FT_CH_L7_L6_Rot_Vit_on_imagenette.py**: Fine-tuning the classifier, layer 7 and layer 6

## Requirements

```
torch
torchvision
transformers
datasets
wandb
huggingface_hub
PIL
numpy
```

## Usage

1. Set the API keys for Weights & Biases and HuggingFace:
```python
WANDB_API_KEY='your_wandb_key'
HF_API_KEY='your_hf_key'
WANDB_USERNAME='your_wandb_username'
```

2. Pre-train the model:
```
python main/single_pre_train_experiments/1_pre_Train_ROT_ViT.py
```

3. Linear probing:
```
python main/single_pre_train_experiments/2a_LP_RotVit_on_Imagenette.py
```

4. Fine-tuning:
```
python main/single_pre_train_experiments/2b_FT_CH_L7_Rot_Vit_on_imagenette.py
python main/single_pre_train_experiments/2c_FT_CH_L7_L6_Rot_Vit_on_imagenette.py
```

5. Feature Analysis:
```
python main/featuresAnalysis.py
```

6. To run the full workflow (advised):
```
python main/main.py
```

## Experiment Tracking

The project uses Weights & Biases (wandb) for experiment tracking. Results are organized into the following projects:
- **vit-self-supervised-rotation**: Pre-training and fine-tuning experiments
- **vit-feature-analysis**: Feature representation analysis

## Results

The experimental results can be viewed via the Weights & Biases dashboard, where the following are tracked:
- Training and validation accuracy
- Training and validation loss
- Learning rate during training
- Comparison between different fine-tuning strategies
- Feature representation analysis across layers

## References

- [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929)
- [Rotation as a Self-Supervised Task](https://arxiv.org/abs/1803.07728)

## Authors

Mencucci Marco, Università degli Studi di Firenze
