import pytest
import torch
from utils.model import DiffusionModel

@pytest.fixture
def model():
    return DiffusionModel(
        num_classes=2,
        embed_dim=128,
        learning_rate=1e-4,
        weight_decay=0.01,
        scheduler_gamma=0.99,
        classification_interval=5,
        test_classification=True,
        enable_classification=True
    )

def test_model_initialization(model):
    """Test if model initializes correctly"""
    assert model.num_classes == 2
    assert model.embed_dim == 128
    assert model.learning_rate == 1e-4
    assert model.weight_decay == 0.01
    assert model.scheduler_gamma == 0.99
    assert model.classification_interval == 5
    assert model.test_classification is True
    assert model.enable_classification is True

def test_model_forward(model):
    """Test if model can process a batch of data"""
    batch_size = 4
    images = torch.randn(batch_size, 3, 32, 32)
    labels = torch.randint(0, 2, (batch_size,))
    
    # Test noise prediction
    noise = torch.randn_like(images)
    timesteps = torch.randint(0, 1000, (batch_size,))
    noisy_images = model.scheduler.add_noise(images, noise, timesteps)
    
    noise_pred = model.model(
        noisy_images,
        timesteps,
        encoder_hidden_states=model.embedding(labels).unsqueeze(1)
    ).sample
    
    assert noise_pred.shape == (batch_size, 3, 32, 32)

def test_classification(model):
    """Test if model can classify images"""
    if not model.enable_classification:
        pytest.skip("Classification is not enabled")
        
    image = torch.randn(3, 32, 32)
    label, errors = model.classify_image(image)
    
    assert isinstance(label, int)
    assert label in [0, 1]
    assert errors.shape == (2,)
    assert errors.min() >= 0 