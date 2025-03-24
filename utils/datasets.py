import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from typing import Iterator, List, Optional, Tuple
import torchvision.transforms as transforms
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
wandb.require("core")  
import pandas as pd
import lightning as L

class TestCifarDataset(Dataset):
    """
    Custom dataset for testing with controlled class distribution.
    """
    def __init__(self, img_size: Tuple[int, int, int], data_len: int, prob_dist: List[float]):
        """
        Args:
            img_size (tuple): Image dimensions (channels, height, width)
            data_len (int): Number of samples in the dataset
            prob_dist (list): Probability distribution for class sampling
        """
        self.img_size = img_size
        self.data_len = data_len
        self.prob_dist = prob_dist
        # Validate probability distribution
        assert abs(sum(prob_dist) - 1.0) < 1e-6, "Probability distribution must sum to 1"
        assert len(prob_dist) >= 2, "Need at least 2 classes"

    def __len__(self) -> int:
        return self.data_len
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # Generate random image tensor
        ret_img = torch.randn(self.img_size)
        # Sample class label according to probability distribution
        list_of_candidates = range(len(self.prob_dist))
        label = np.random.choice(list_of_candidates, 1, p=self.prob_dist)[0]
        return ret_img, label

class DebuggingSampler(Sampler):
    """
    Sampler for debugging that only returns a small number of samples.
    """
    def __init__(self, n: int = 10):
        self.n = n

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.n))

    def __len__(self) -> int:
        return self.n

class DiffusionDataModule(L.LightningDataModule):
    """
    Data module for handling CIFAR-10 dataset with binary classification.
    """
    def __init__(self, batch_size: int = 64, 
                 num_workers: int = 4, 
                 binary_classes: bool = True,
                 debug: bool = False):
        """
        Args:
            batch_size (int): Batch size for dataloaders
            num_workers (int): Number of workers for data loading
            binary_classes (bool): If True, convert to binary classification (airplane vs. not airplane)
            debug (bool): If True, use debugging sampler
        """
        super().__init__()  # Call the parent constructor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.binary_classes = binary_classes
        self.debug = debug
        
        self.augment = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def convert_to_binary_label(self, label: int) -> int:
        """Convert CIFAR-10 labels to binary (airplane vs. not airplane)"""
        return 0 if label == 0 else 1  # 0: Airplane, 1: Not Airplane

    def preprocess_data(self, examples: dict) -> dict:
        """Preprocess data with augmentations and label conversion"""
        images = examples["img"]
        examples["images"] = [self.augment(image) for image in images]
        labels = examples["label"]
        
        if self.binary_classes:
            examples["labels"] = [self.convert_to_binary_label(label) for label in labels]
        else:
            examples["labels"] = labels
            
        return examples

    def preprocess_test(self, examples: dict) -> dict:
        """Preprocess test data without augmentations"""
        images = examples["img"]
        examples["images"] = [self.test_transform(image) for image in images]
        labels = examples["label"]
        
        if self.binary_classes:
            examples["labels"] = [self.convert_to_binary_label(label) for label in labels]
        else:
            examples["labels"] = labels
            
        return examples

    def prepare_data(self):
        """Download the dataset if needed"""
        load_dataset("cifar10")

    def setup(self, stage=None):
        """Set up datasets - required by LightningDataModule"""
        if stage == "fit" or stage is None:
            self.train_dataset = self.get_train_dataset()
            self.val_dataset = self.get_test_dataset()
        if stage == "test" or stage is None:
            self.test_dataset = self.get_test_dataset()

    def get_train_dataset(self):
        """Get preprocessed training dataset"""
        dataset = load_dataset("cifar10")
        return dataset["train"].with_transform(self.preprocess_data)
    
    def get_test_dataset(self):
        """Get preprocessed test dataset"""
        dataset = load_dataset("cifar10")
        return dataset["test"].with_transform(self.preprocess_test)

    def train_dataloader(self):
        """Get training dataloader - required by LightningDataModule"""
        dataset = self.get_train_dataset()
        
        if self.debug:
            sampler = DebuggingSampler(10)
            shuffle = False
        else:
            sampler = None
            shuffle = True
            
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers, 
            collate_fn=self.collate_fn,
            sampler=sampler,
            shuffle=shuffle
        )

    def val_dataloader(self):
        """Get validation dataloader - required by LightningDataModule"""
        dataset = self.get_test_dataset()
        
        if self.debug:
            sampler = DebuggingSampler(10)
            shuffle = False
        else:
            sampler = None
            shuffle = False
            
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers, 
            collate_fn=self.collate_fn,
            sampler=sampler,
            shuffle=shuffle
        )

    def test_dataloader(self):
        """Get test dataloader - required by LightningDataModule"""
        # Use the same dataset as validation for now
        return self.val_dataloader()

    def collate_fn(self, batch):
        """Collate function for dataloaders"""
        images = torch.stack([item["images"] for item in batch])
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return {"images": images, "labels": labels}

    def get_test_dataloader(self, prob_dist=None):
        """Get custom test dataloader with controlled class distribution"""
        if prob_dist is None:
            if self.binary_classes:
                prob_dist = [0.5, 0.5]  # Equal distribution for binary classes
            else:
                prob_dist = [0.1] * 10  # Equal distribution for 10 classes
                
        img_size = (3, 32, 32)
        data_len = 30
        dataset = TestCifarDataset(img_size, data_len, prob_dist)
        
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )

def evaluate_classifier(model, dataloader, device, num_classes=2, num_samples=None):
    """
    Evaluate the diffusion model as a classifier.
    
    Args:
        model: Diffusion model with classification capability
        dataloader: DataLoader for evaluation
        device: Torch device
        num_classes: Number of classes
        num_samples: Maximum number of samples to evaluate
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    true_labels = []
    pred_labels = []
    confidence_scores = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            labels = batch["labels"].to(device)
            
            # Get predictions and confidence scores
            preds, errors = model.classify_batch(images, num_classes=num_classes)
            
            # Store results
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())
            
            # Calculate confidence scores (inverse of minimum error)
            for i, pred in enumerate(preds):
                inverse_errors = 1.0 / (errors[i] + 1e-6)
                confidence = inverse_errors[pred.item()] / torch.sum(inverse_errors)
                confidence_scores.append(confidence.item())
            
            sample_count += len(images)
            if num_samples is not None and sample_count >= num_samples:
                break
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')
    conf_matrix = confusion_matrix(true_labels, pred_labels)
    
    # Calculate AUC-ROC for binary classification
    roc_auc = None
    if num_classes == 2:
        # Convert confidence scores to probabilities for class 1
        probs = []
        for i, pred in enumerate(pred_labels):
            if pred == 1:
                probs.append(confidence_scores[i])
            else:
                probs.append(1 - confidence_scores[i])
        try:
            roc_auc = roc_auc_score(true_labels, probs)
        except:
            roc_auc = None
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'roc_auc': roc_auc,
        'sample_count': sample_count,
        'true_labels': true_labels,
        'pred_labels': pred_labels,
        'confidence_scores': confidence_scores
    }
    
    return metrics

def log_classification_metrics(metrics, epoch=None, prefix="val"):
    """
    Log classification metrics to wandb.
    
    Args:
        metrics: Dictionary of evaluation metrics
        epoch: Current epoch number
        prefix: Prefix for metric names
    """
    # Check if wandb is initialized
    if not wandb.run:
        return
        
    log_dict = {
        f"{prefix}/accuracy": metrics['accuracy'],
        f"{prefix}/precision": metrics['precision'],
        f"{prefix}/recall": metrics['recall'],
        f"{prefix}/f1": metrics['f1'],
    }
    
    if metrics['roc_auc'] is not None:
        log_dict[f"{prefix}/roc_auc"] = metrics['roc_auc']
    
    if epoch is not None:
        log_dict["epoch"] = epoch
    
    wandb.log(log_dict)
    
    # Log confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{prefix.capitalize()} Confusion Matrix')
    wandb.log({f"{prefix}/confusion_matrix": wandb.Image(plt)})
    plt.close()

def visualize_samples(original_images, generated_images, labels, num_classes=2, nrow=4):
    """
    Visualize original and generated samples side by side.
    
    Args:
        original_images: Tensor of original images
        generated_images: Tensor of generated images
        labels: Tensor of image labels
        num_classes: Number of classes
        nrow: Number of images per row in the grid
    
    Returns:
        fig: Matplotlib figure
    """
    # Denormalize images from [-1, 1] to [0, 1]
    original_images = (original_images * 0.5) + 0.5
    generated_images = (generated_images * 0.5) + 0.5
    
    # Create a grid of images
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Plot original images
    axes[0].set_title("Original Images")
    for i in range(min(nrow, len(original_images))):
        ax = fig.add_subplot(2, nrow, i + 1)
        img = original_images[i].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.set_title(f"Class {labels[i].item()}")
        ax.axis('off')
    
    # Plot generated images
    axes[1].set_title("Generated Images")
    for i in range(min(nrow, len(generated_images))):
        ax = fig.add_subplot(2, nrow, i + nrow + 1)
        img = generated_images[i].permute(1, 2, 0).cpu().numpy()
        ax.imshow(img)
        ax.set_title(f"Class {labels[i].item()}")
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def plot_training_metrics(metrics_history):
    """
    Plot training metrics over time.
    
    Args:
        metrics_history: Dictionary of metric histories
    
    Returns:
        figs: Dictionary of matplotlib figures
    """
    figs = {}
    
    # Plot accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics_history['epochs'], metrics_history['train_accuracy'], label='Train')
    ax.plot(metrics_history['epochs'], metrics_history['val_accuracy'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy')
    ax.legend()
    ax.grid(True)
    figs['accuracy'] = fig
    
    # Plot loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics_history['epochs'], metrics_history['train_loss'], label='Train')
    ax.plot(metrics_history['epochs'], metrics_history['val_loss'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True)
    figs['loss'] = fig
    
    # Plot AUC-ROC if available
    if 'val_roc_auc' in metrics_history and any(x is not None for x in metrics_history['val_roc_auc']):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(metrics_history['epochs'], metrics_history['val_roc_auc'], label='Validation')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC-ROC')
        ax.set_title('ROC AUC Score')
        ax.legend()
        ax.grid(True)
        figs['roc_auc'] = fig
    
    return figs

def create_thesis_tables(metrics_history):
    """
    Create tables suitable for a thesis.
    
    Args:
        metrics_history: Dictionary of metric histories
    
    Returns:
        tables: Dictionary of pandas DataFrames
    """
    tables = {}
    
    # Overall metrics table
    overall_metrics = pd.DataFrame({
        'Epoch': metrics_history['epochs'],
        'Train Loss': metrics_history['train_loss'],
        'Val Loss': metrics_history['val_loss'],
        'Train Accuracy': metrics_history['train_accuracy'],
        'Val Accuracy': metrics_history['val_accuracy'],
    })
    
    if 'val_roc_auc' in metrics_history:
        overall_metrics['Val ROC-AUC'] = metrics_history['val_roc_auc']
    
    tables['overall_metrics'] = overall_metrics
    
    # Best metrics table
    best_epoch = overall_metrics['Val Accuracy'].idxmax()
    best_metrics = pd.DataFrame({
        'Metric': ['Best Epoch', 'Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy'],
        'Value': [
            metrics_history['epochs'][best_epoch],
            metrics_history['train_loss'][best_epoch],
            metrics_history['val_loss'][best_epoch],
            metrics_history['train_accuracy'][best_epoch],
            metrics_history['val_accuracy'][best_epoch],
        ]
    })
    
    if 'val_roc_auc' in metrics_history and metrics_history['val_roc_auc'][best_epoch] is not None:
        best_metrics = pd.concat([
            best_metrics,
            pd.DataFrame({'Metric': ['Val ROC-AUC'], 'Value': [metrics_history['val_roc_auc'][best_epoch]]})
        ])
    
    tables['best_metrics'] = best_metrics
    
    return tables