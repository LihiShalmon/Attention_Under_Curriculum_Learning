<placeholder for readme file>
# CNN with Attention: Experiments with Curriculum Learning

## Overview
This project investigates the impact of curriculum learning on Convolutional Neural Networks (CNNs), CNNs with attention mechanisms, and pure Transformer architectures. Our study explores how structuring the order of training samples influences model convergence, performance, and data efficiency.

## Repository Structure
```
|-- data/                # Dataset files and preprocessing scripts
|-- models/              # Model architectures (CNN, CNN+Attention, Transformer)
|-- experiments/         # Experiment scripts and configurations
|-- results/             # Experimental results, logs, and model checkpoints
|-- utils/               # Utility functions for training, evaluation, and visualization
|-- notebooks/           # Jupyter notebooks for exploratory data analysis
|-- README.md            # Project documentation
|-- requirements.txt     # Dependencies
|-- train.py             # Main training script
|-- evaluate.py          # Evaluation script
|-- config.yaml          # Configuration file for hyperparameters
```

## Installation
To set up the environment, install the dependencies using:
```bash
pip install -r requirements.txt
```
Ensure that the necessary dataset is available in the `data/` directory.

## Dataset
We use **CIFAR-100 Mammals**, a subset of CIFAR-100 containing five classes and 3,000 images. The dataset is preprocessed using a standard augmentation pipeline:
- Training: Random cropping + horizontal flipping
- Testing: Normalization and tensor conversion

## Curriculum Learning Strategy
Our curriculum learning approach consists of three components:
1. **Scoring Function**: Assigns difficulty scores to samples. We use [TBD model] for this.
2. **Ordering Strategy**: Experiments include curriculum (easy-to-hard), anti-curriculum (hard-to-easy), and random ordering.
3. **Pacing Function**: Controls the rate of sample introduction per epoch. We implement exponential pacing:
   \[ g_\theta(i) = N_0 \times \left( \frac{N}{N_0} \right)^{\frac{i}{M}} \]

## Model Architectures
We experiment with three model architectures:
1. **CNN (ResNet-50)**: Traditional convolutional model.
2. **CNN with Attention (CBAM)**: Adds spatial and channel-wise attention to ResNet.
3. **Vision Transformer (ViT)**: Self-attention-based model for vision tasks.

## Training and Evaluation
Run training using:
```bash
python train.py --config config.yaml
```
Evaluate trained models with:
```bash
python evaluate.py --model <model_path>
```

## Hyperparameters
| Hyperparameter       | Value                  |
|----------------------|------------------------|
| Batch Size          | 128                     |
| Epochs              | 182                     |
| Learning Rate       | 0.1                     |
| Scheduler           | Milestones: 91, 136 (gamma=0.1) |
| Optimizer           | SGD (momentum=0.9)      |
| Weight Decay        | 0.0001                  |

## Results
(TBD: Include key findings, performance metrics, and comparisons with prior work.)

## Future Work
- Extend experiments to larger datasets.
- Explore additional curriculum designs.
- Investigate hybrid models integrating CNN feature extractors with attention mechanisms.

## Contact
For questions or contributions, contact:
- **First Author**: first.author@domain.com
- **Second Author**: second.author@domain.com

---
*This README is a work in progress. Some details are placeholders and will be updated as experiments progress.*

