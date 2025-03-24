import os
import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from utils.datasets import visualize_samples, log_classification_metrics, DiffusionDataModule
from utils.model import DiffusionModel  
import glob
import time

def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3):
    """Clean up old checkpoints, keeping only the N most recent ones."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if len(checkpoint_files) <= keep_last_n:
        return
        
    # Sort files by modification time
    checkpoint_files.sort(key=lambda x: os.path.getmtime(x))
    
    # Remove older files
    for file in checkpoint_files[:-keep_last_n]:
        try:
            os.remove(file)
            print(f"Removed old checkpoint: {file}")
        except Exception as e:
            print(f"Error removing checkpoint {file}: {e}")

class CheckpointCleanupCallback(L.Callback):
    """Callback to clean up old checkpoints periodically"""
    def __init__(self, checkpoint_dir, keep_last_n=3):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        
    def on_validation_epoch_end(self, trainer, pl_module):
        cleanup_old_checkpoints(self.checkpoint_dir, self.keep_last_n)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion model for image generation and classification")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to store datasets")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--binary_classes", action="store_true", help="Use binary classification (airplane vs. not airplane)")
    
    # Model parameters
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes (2 for binary, 10 for full CIFAR-10)")
    parser.add_argument("--embed_dim", type=int, default=128, help="Dimension of class embeddings")
    
    # Training parameters
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--scheduler_gamma", type=float, default=0.99, help="Gamma for learning rate scheduler")
    parser.add_argument("--classification_interval", type=int, default=10, 
                        help="Interval (in epochs) to run classification evaluation")
    
    # System parameters
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--precision", type=str, default="16-mixed", 
                        help="Precision for training (16-mixed, 32-true, etc.)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Logging and checkpointing
    parser.add_argument("--project_name", type=str, default="diffusion-classifier", help="Project name for wandb")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for wandb")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Debug mode
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with minimal data")
    parser.add_argument("--test_classification", action="store_true", default=True, 
                        help="Test classification during validation")
    parser.add_argument("--fast_dev_run", action="store_true", help="Do a fast dev run with 1 batch each")
    
    parser.add_argument("--storage_dir", type=str, default="./outputs", help="Directory to save WandB Logs")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    L.seed_everything(args.seed)
    
    # Ensure checkpoint directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.num_classes is None:
        args.num_classes = 2 if args.binary_classes else 10
    # Set up data module
    data_module = DiffusionDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        binary_classes=args.binary_classes,
        debug=args.debug
    )
    
    # Adjust num_classes based on binary_classes if not explicitly set
    if args.num_classes is None:
        args.num_classes = 2 if args.binary_classes else 10
    
    # Set up model
    model = DiffusionModel(
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_gamma=args.scheduler_gamma,
        classification_interval=args.classification_interval,
        test_classification=args.test_classification,
        enable_classification=True
    )
    
    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            save_top_k=1,  # Only save the best model
            mode="min",
            save_last=False,  # Don't save last checkpoint
            every_n_epochs=10  # Save checkpoints less frequently
        ),
        LearningRateMonitor(logging_interval="epoch"),
        CheckpointCleanupCallback(args.checkpoint_dir)
    ]

    # Set up wandb logger with reduced logging
    logger = WandbLogger(
        project=args.project_name,
        name=args.run_name,
        save_dir=args.storage_dir,
        log_model=None  # Disable model saving to wandb
    )
    
    # Log hyperparameters
    logger.log_hyperparams(vars(args))
    
    # Set up trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else None,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        deterministic=True,
        check_val_every_n_epoch=5, 
        log_every_n_steps=1000  
    )

    
    # Train model
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_from_checkpoint)

    
    # Final evaluation
    if not args.fast_dev_run and not args.debug:
        print("Running final evaluation...")
        trainer.validate(model, data_module)
        
        # Generate samples and save metrics ONLY on rank 0
        if trainer.global_rank == 0:
            try:
                # Generate samples for each class
                device = next(model.parameters()).device
                num_samples = min(10, args.num_classes)
                labels = torch.arange(num_samples, device=device)
                
                print(f"Generating {num_samples} samples...")
                generated_images = model.sample_images(labels)
                
                # Save generated images
                import torchvision
                grid = torchvision.utils.make_grid(
                    generated_images.cpu().clamp(-1, 1) * 0.5 + 0.5,
                    nrow=5
                )
                torchvision.utils.save_image(grid, "final_samples.png")
                print("Saved generated samples to final_samples.png")
                
                # Get metrics history and export
                metrics_history = model.get_metrics_history()
                if metrics_history is not None:
                    import pandas as pd
                    # Clean up metrics history by removing None values
                    cleaned_metrics = {}
                    max_len = max(len(v) for v in metrics_history.values())
                    
                    # Ensure all arrays have the same length
                    for key, values in metrics_history.items():
                        if len(values) < max_len:
                            values.extend([None] * (max_len - len(values)))
                        cleaned_metrics[key] = values
                    
                    # Create DataFrame and save
                    df = pd.DataFrame(cleaned_metrics)
                    df.to_csv("metrics_history.csv", index=False)
                    print("Saved metrics history to metrics_history.csv")
            except Exception as e:
                print(f"Warning: Error during final evaluation on rank 0: {str(e)}")
    
    print("Training complete!")

if __name__ == "__main__":
    main()