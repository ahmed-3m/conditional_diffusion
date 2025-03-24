import torch
import diffusers
import torch.nn.functional as F
import torchvision
import wandb
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Dict, List, Optional, Tuple, Any
from utils.datasets import evaluate_classifier, log_classification_metrics

class DiffusionModel(L.LightningModule):
    def __init__(
        self, 
        num_classes: int = 2, 
        embed_dim: int = 128,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        scheduler_gamma: float = 0.99,
        classification_interval: int = 10,
        test_classification: bool = True,
        enable_classification: bool = True
    ):
        """
        Diffusion model with built-in classification capability.
        
        Args:
            num_classes: Number of classes for conditioning
            embed_dim: Dimension of class embeddings
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            scheduler_gamma: Gamma value for learning rate scheduler
            classification_interval: Interval (in epochs) to run classification evaluation
            test_classification: Whether to run classification evaluation during validation
            enable_classification: Whether to enable classification capabilities
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Model configuration
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_gamma = scheduler_gamma
        self.classification_interval = classification_interval
        self.test_classification = test_classification
        self.enable_classification = enable_classification
        
        # Initialize metrics history with empty lists
        self.metrics_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'val_roc_auc': [],
        }
        
        # Configure UNet for cross-attention with label conditioning
        self.model = diffusers.UNet2DConditionModel(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            cross_attention_dim=embed_dim,
            down_block_types=(
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
            ),
            layers_per_block=2,
            block_out_channels=(128, 256, 256, 512),
        )
        
        # Noise scheduler
        self.scheduler = diffusers.schedulers.DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        # Class embedding layer
        self.embedding = torch.nn.Embedding(num_classes, embed_dim)

    def batch_step(self, batch, batch_idx):
        """Single batch step for training and validation"""
        images, labels = batch["images"], batch["labels"]
        
        # Sample random noise
        noise = torch.randn_like(images)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, 
            (images.shape[0],), 
            device=self.device
        )
        
        # Add noise to images according to timesteps
        noisy_images = self.scheduler.add_noise(images, noise, timesteps)
        
        # Convert class labels to embeddings
        label_embeddings = self.embedding(labels).unsqueeze(1)
        
        # Predict noise using the model
        noise_pred = self.model(
            noisy_images, 
            timesteps, 
            encoder_hidden_states=label_embeddings
        ).sample
        
        # Compute loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss, noise_pred, timesteps, noisy_images

    def training_step(self, batch, batch_idx):
        """Training step"""
        loss, _, _, _ = self.batch_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        val_loss, _, _, _ = self.batch_step(batch, batch_idx)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        
        # Store batch for end of validation visualizations
        if batch_idx == 0:
            self.val_batch = batch
            
        return val_loss

    def on_validation_epoch_end(self):
        """End of validation epoch"""
        if not self.test_classification or not self.enable_classification:
            return
        
        # Store metrics history
        self.metrics_history['epochs'].append(self.current_epoch)
        
        # Get current metrics, defaulting to None if not available
        train_loss = self.trainer.callback_metrics.get('train_loss', None)
        val_loss = self.trainer.callback_metrics.get('val_loss', None)
        
        # Store None or float values for losses
        self.metrics_history['train_loss'].append(
            train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss
        )
        self.metrics_history['val_loss'].append(
            val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss
        )
        
        # Only run classification every classification_interval epochs
        if self.current_epoch % self.classification_interval == 0:
            # Only proceed on the main process (rank 0)
            if self.trainer.global_rank == 0:
                # Get the validation dataloader
                val_dataloader = self.trainer.datamodule.val_dataloader()
                
                # Evaluate on a subset of the validation set
                metrics = evaluate_classifier(
                    self, 
                    val_dataloader, 
                    self.device, 
                    num_classes=self.num_classes,
                    num_samples=200  # Limit to 200 samples for faster evaluation
                )
                
                # Store classification metrics
                self.metrics_history['val_accuracy'].append(metrics['accuracy'])
                self.metrics_history['val_roc_auc'].append(metrics['roc_auc'])
                
                # Log metrics only if we have wandb initialized
                if wandb.run:
                    log_classification_metrics(metrics, self.current_epoch, prefix="val")
                    
                # Generate samples for visualization
                if hasattr(self, 'val_batch'):
                    images = self.val_batch["images"][:8]
                    labels = self.val_batch["labels"][:8]
                    
                    # Generate samples
                    labels = labels.to(self.device)
                    generated_images = self.sample_images(labels)
                    
                    # Visualize samples
                    from utils.datasets import visualize_samples
                    fig = visualize_samples(images, generated_images, labels, self.num_classes)
                    if wandb.run:
                        wandb.log({"val/samples": wandb.Image(fig)})
                    
                    # Log generated samples grid
                    grid = torchvision.utils.make_grid(
                        generated_images.clamp(-1, 1) * 0.5 + 0.5,
                        nrow=4
                    )
                    if wandb.run:
                        wandb.log({"val/generated_grid": wandb.Image(grid)})
        else:
            # If we skip classification this epoch, append None to maintain list lengths
            self.metrics_history['val_accuracy'].append(None)
            self.metrics_history['val_roc_auc'].append(None)

    @torch.no_grad()
    def sample_images(self, labels, num_inference_steps=50):
        """
        Generate images using the full denoising process.
        
        Args:
            labels: Tensor of class labels
            num_inference_steps: Number of denoising steps
            
        Returns:
            Tensor of generated images
        """
        # Ensure labels are on the correct device
        labels = labels.to(self.device)
        batch_size = labels.shape[0]
        
        # Start from pure noise
        sample = torch.randn(batch_size, 3, 32, 32, device=self.device)
        
        # Set up denoising schedule
        self.scheduler.set_timesteps(num_inference_steps)
        
        # Move scheduler's internal tensors to the correct device
        self.scheduler.betas = self.scheduler.betas.to(self.device)
        self.scheduler.alphas = self.scheduler.alphas.to(self.device)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        
        # Prepare label embeddings
        label_embeddings = self.embedding(labels).unsqueeze(1)
        
        # Iteratively denoise
        for i, t in enumerate(self.scheduler.timesteps):
            # Expand timestep to batch dimension
            timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model(
                sample, 
                timesteps, 
                encoder_hidden_states=label_embeddings
            ).sample
            
            # Ensure all tensors are on the same device
            noise_pred = noise_pred.to(self.device)
            sample = sample.to(self.device)
            
            # Apply denoising step
            sample = self.scheduler.step(noise_pred, t, sample).prev_sample
            
        return sample

    @torch.no_grad()
    def classify_image(self, image, num_classes=None, num_trials=10):
        """
        Classify a single image using the diffusion model.
        
        Args:
            image: Image tensor of shape [3, 32, 32]
            num_classes: Number of classes to consider
            num_trials: Number of random trials to average over
            
        Returns:
            label: Predicted class label
            errors: Error scores for each class
        """
        if not self.enable_classification:
            raise RuntimeError("Classification is not enabled for this model")
            
        if num_classes is None:
            num_classes = self.num_classes
            
        # Move image to device and add batch dimension
        image = image.to(self.device).unsqueeze(0)
        
        # Initialize errors
        errors = torch.zeros(num_classes, device=self.device)
        
        # Run multiple trials and average
        for _ in range(num_trials):
            # Sample random timestep
            t = torch.randint(
                0, self.scheduler.config.num_train_timesteps, 
                (1,), 
                device=self.device
            )
            
            # Generate random noise
            noise = torch.randn_like(image)
            
            # Add noise to image
            noisy_image = self.scheduler.add_noise(image, noise, t)
            
            # Repeat the noisy image for each class
            noisy_images = noisy_image.repeat(num_classes, 1, 1, 1)
            
            # Repeat the timestep for each class
            timesteps = t.repeat(num_classes)
            
            # Create embeddings for all potential classes
            labels = torch.arange(num_classes, device=self.device)
            label_embeddings = self.embedding(labels).unsqueeze(1)
            
            # Predict noise for each class
            noise_preds = self.model(
                noisy_images,
                timesteps,
                encoder_hidden_states=label_embeddings
            ).sample
            
            # Compute MSE between predicted and true noise
            noise_expanded = noise.repeat(num_classes, 1, 1, 1)
            mse = F.mse_loss(noise_preds, noise_expanded, reduction='none').mean(dim=[1, 2, 3])
            
            # Accumulate errors
            errors += mse
            
        # Average errors over trials
        avg_errors = errors / num_trials
        
        # Return prediction and errors
        return avg_errors.argmin().item(), avg_errors.cpu().numpy()

    @torch.no_grad()
    def classify_batch(self, images, num_classes=None, num_trials=10):
        """
        Classify a batch of images using the diffusion model.
        
        Args:
            images: Batch of images of shape [B, 3, 32, 32]
            num_classes: Number of classes to consider
            num_trials: Number of random trials to average over
            
        Returns:
            labels: Tensor of predicted class labels
            errors: Tensor of error scores for each image and class
        """
        if not self.enable_classification:
            raise RuntimeError("Classification is not enabled for this model")
            
        if num_classes is None:
            num_classes = self.num_classes
            
        batch_size = images.shape[0]
        predictions = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        all_errors = torch.zeros(batch_size, num_classes, device=self.device)
        
        # Process each image individually
        for i, image in enumerate(images):
            _, errors = self.classify_image(image, num_classes, num_trials)
            errors_tensor = torch.tensor(errors, device=self.device)
            predictions[i] = errors_tensor.argmin()
            all_errors[i] = errors_tensor
            
        return predictions, all_errors

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=1, 
            gamma=self.scheduler_gamma
        )
        
        return [optimizer], [scheduler]

    def get_metrics_history(self):
        """Get the metrics history dictionary"""
        return self.metrics_history