# Conditional Diffusion Classifier

A PyTorch implementation of a conditional diffusion model that can both generate images and perform classification. The model is trained on CIFAR-10 dataset with a focus on binary classification (airplane vs. not airplane).

## Features

- Conditional image generation using diffusion models
- Built-in classification capability
- Binary classification support (airplane vs. not airplane)
- Multi-GPU training support
- Weights & Biases integration for experiment tracking
- Automatic checkpoint management
- Comprehensive evaluation metrics

## Project Structure

```
.
├── main.py                 # Main training script
├── run_training.sh         # Training script with default parameters
├── utils/
│   ├── __init__.py
│   ├── datasets.py        # Dataset handling and evaluation utilities
│   └── model.py          # Diffusion model implementation
├── checkpoints/           # Model checkpoints (not tracked in git)
└── wandb/                # Weights & Biases logs (not tracked in git)
```

## Requirements

- Python 3.8+
- PyTorch
- Lightning
- diffusers
- datasets
- wandb
- scikit-learn
- matplotlib
- seaborn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ahmed-3m/conditional_diffusion.git
cd conditional_diffusion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

To train the model with default parameters:

```bash
bash run_training.sh
```

Or customize training parameters:

```bash
python main.py \
    --batch_size 64 \
    --num_workers 4 \
    --binary_classes \
    --num_classes 2 \
    --embed_dim 128 \
    --max_epochs 801 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --scheduler_gamma 0.99 \
    --classification_interval 5 \
    --gpus 2 \
    --precision "16-mixed" \
    --seed 42 \
    --project_name "diffusion-cifar" \
    --run_name "your-run-name"
```

### Key Parameters

- `--binary_classes`: Enable binary classification (airplane vs. not airplane)
- `--num_classes`: Number of classes (2 for binary, 10 for full CIFAR-10)
- `--embed_dim`: Dimension of class embeddings
- `--classification_interval`: Interval (in epochs) to run classification evaluation
- `--gpus`: Number of GPUs to use
- `--precision`: Training precision (16-mixed, 32-true)

### Debug Mode

For quick testing, use debug mode:

```bash
python main.py --debug --fast_dev_run --batch_size 16 --max_epochs 2
```

## Model Architecture

The model uses a UNet architecture with cross-attention for conditional generation. Key components:

- UNet2DConditionModel with cross-attention blocks
- Class embeddings for conditioning
- DDPMScheduler for noise scheduling
- Built-in classification capability using noise prediction error

## Evaluation

The model is evaluated on:
- Generation quality (visual samples)
- Classification accuracy
- ROC-AUC score
- Precision, Recall, and F1 score
- Confusion matrix

Results are logged to Weights & Biases and saved locally in:
- `metrics_history.csv`: Training metrics over time
- `final_samples.png`: Generated samples at the end of training

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{conditional_diffusion,
  author = {Ahmed Mohammed},
  title = {Conditional Diffusion Classifier},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/ahmed-3m/conditional_diffusion}
}
``` 