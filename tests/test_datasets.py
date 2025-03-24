import pytest
import torch
from utils.datasets import DiffusionDataModule, TestCifarDataset

@pytest.fixture
def data_module():
    return DiffusionDataModule(
        batch_size=4,
        num_workers=0,
        binary_classes=True,
        debug=True
    )

def test_data_module_initialization(data_module):
    """Test if data module initializes correctly"""
    assert data_module.batch_size == 4
    assert data_module.num_workers == 0
    assert data_module.binary_classes is True
    assert data_module.debug is True

def test_binary_label_conversion(data_module):
    """Test binary label conversion"""
    # Test airplane class (0)
    assert data_module.convert_to_binary_label(0) == 0
    # Test non-airplane class (1)
    assert data_module.convert_to_binary_label(1) == 1

def test_test_dataset():
    """Test the test dataset creation"""
    img_size = (3, 32, 32)
    data_len = 10
    prob_dist = [0.5, 0.5]
    
    dataset = TestCifarDataset(img_size, data_len, prob_dist)
    
    assert len(dataset) == data_len
    
    # Test sample
    image, label = dataset[0]
    assert image.shape == img_size
    assert label in [0, 1]

def test_data_loading(data_module):
    """Test if data can be loaded"""
    # Setup the data module
    data_module.setup()
    
    # Get a batch from the train dataloader
    train_dataloader = data_module.train_dataloader()
    batch = next(iter(train_dataloader))
    
    assert "images" in batch
    assert "labels" in batch
    assert batch["images"].shape[0] <= data_module.batch_size
    assert batch["labels"].shape[0] <= data_module.batch_size 