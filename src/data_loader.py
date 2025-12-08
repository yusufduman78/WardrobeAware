"""
DeepFashion Dataset Data Loader for Fashion Recommendation System

This module provides comprehensive data loading capabilities for the DeepFashion dataset,
including proper train/validation splits, data augmentation, and performance monitoring.

Author: isobed18
Date: 2025-10-04
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, datasets
from PIL import Image
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
import time
from pathlib import Path
import json


class DeepFashionDataset(Dataset):
    """
    Custom dataset class for DeepFashion dataset with proper train/validation splits.
    
    The dataset should be organized as:
    newdataset/
    ├── train/
    │   ├── 1/  # category 1
    │   │   ├── image1.jpg
    │   │   └── image2.jpg
    │   └── 2/  # category 2
    └── val/
        ├── 1/
        └── 2/
    
    This class handles loading images, applying transforms, and managing metadata
    for efficient training and validation.
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        load_metadata: bool = True
    ):
        """
        Initialize DeepFashion dataset.
        
        Args:
            root_dir: Path to the dataset root directory
            split: 'train' or 'val'
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on targets
            load_metadata: Whether to load and store metadata about the dataset
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.load_metadata = load_metadata
        
        # Validate split
        if split not in ['train', 'val']:
            raise ValueError(f"Split must be 'train' or 'val', got {split}")
        
        # Load dataset metadata
        self.images, self.labels, self.classes = self._load_dataset_metadata()
        
        if load_metadata:
            self.metadata = self._generate_metadata()
        else:
            self.metadata = None
    
    def _load_dataset_metadata(self) -> Tuple[List[str], List[int], List[str]]:
        """Load image paths, labels, and class names from the dataset."""
        split_dir = self.root_dir / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        images = []
        labels = []
        classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        
        # Create label to index mapping
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        # Collect all image paths and their labels
        for class_name in classes:
            class_dir = split_dir / class_name
            if class_dir.is_dir():
                for img_file in class_dir.iterdir():
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        images.append(str(img_file))
                        labels.append(class_to_idx[class_name])
        
        return images, labels, classes
    
    def _generate_metadata(self) -> Dict:
        """Generate comprehensive metadata about the dataset."""
        metadata = {
            'split': self.split,
            'total_images': len(self.images),
            'num_classes': len(self.classes),
            'classes': self.classes,
            'class_distribution': {},
            'image_paths': self.images,
            'labels': self.labels
        }
        
        # Calculate class distribution
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            metadata['class_distribution'][self.classes[label]] = int(count)
        
        return metadata
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.
        
        Loads the image at the given index, applies transforms, and returns
        the processed image tensor along with its class label.
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets.
        
        Computes inverse frequency weights to help the model learn from
        underrepresented classes more effectively.
        """
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        total_samples = len(self.labels)
        
        # Calculate inverse frequency weights
        weights = total_samples / (len(unique_labels) * counts)
        class_weights = torch.zeros(len(self.classes))
        
        for label, weight in zip(unique_labels, weights):
            class_weights[label] = weight
        
        return class_weights
    
    def save_metadata(self, filepath: str) -> None:
        """
        Save dataset metadata to a JSON file.
        
        Useful for debugging and analysis - saves all dataset statistics
        and class distributions to a file for later inspection.
        """
        if self.metadata is None:
            raise ValueError("Metadata not loaded. Set load_metadata=True during initialization.")
        
        with open(filepath, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def print_summary(self) -> None:
        """
        Print a summary of the dataset.
        
        Displays key statistics including total images, number of classes,
        and class distribution for quick dataset inspection.
        """
        print(f"\n=== DeepFashion Dataset Summary ({self.split}) ===")
        print(f"Total images: {len(self.images)}")
        print(f"Number of classes: {len(self.classes)}")
        print(f"Classes: {self.classes}")
        print(f"Class distribution:")
        for class_name, count in self.metadata['class_distribution'].items():
            print(f"  {class_name}: {count}")


def get_data_transforms(
    input_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    augmentation_strength: str = 'medium'
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get data transforms for training and validation.
    
    Creates appropriate image transformations for both training and validation.
    Training transforms include augmentation, while validation transforms are minimal.
    
    Args:
        input_size: Size to resize images to (typically 224 for ImageNet pretrained models)
        mean: Normalization mean values (ImageNet standard)
        std: Normalization std values (ImageNet standard)
        augmentation_strength: 'light', 'medium', or 'heavy' - controls augmentation intensity
    
    Returns:
        Tuple of (train_transforms, val_transforms)
    """
    
    # Base transforms
    base_transforms = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]
    
    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose(base_transforms)
    
    # Training transforms with augmentation
    if augmentation_strength == 'light':
        train_transforms = transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    elif augmentation_strength == 'medium':
        train_transforms = transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    elif augmentation_strength == 'heavy':
        train_transforms = transforms.Compose([
            transforms.Resize((input_size + 64, input_size + 64)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    else:
        raise ValueError(f"Unknown augmentation_strength: {augmentation_strength}")
    
    return train_transforms, val_transforms


def create_data_loaders(
    dataset_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    input_size: int = 224,
    augmentation_strength: str = 'medium',
    persistent_workers: bool = True,
    balance_classes: bool = False
) -> Tuple[DataLoader, DataLoader, DeepFashionDataset, DeepFashionDataset]:
    """
    Create training and validation data loaders.
    
    This is the main function to set up data loading for training. It creates
    both training and validation datasets with appropriate transforms and
    optimizes the data loading pipeline for performance.
    
    Args:
        dataset_root: Path to the dataset root directory
        batch_size: Batch size for data loaders (adjust based on GPU memory)
        num_workers: Number of worker processes for data loading (typically 4-8)
        pin_memory: Whether to pin memory for faster GPU transfer (recommended: True)
        input_size: Size to resize images to (224 for most pretrained models)
        augmentation_strength: Strength of data augmentation ('light', 'medium', 'heavy')
        persistent_workers: Whether to keep workers alive between epochs (recommended: True)
    
    Returns:
        Tuple of (train_loader, val_loader, train_dataset, val_dataset)
    """
    
    # Get transforms
    train_transforms, val_transforms = get_data_transforms(
        input_size=input_size,
        augmentation_strength=augmentation_strength
    )
    
    # Create datasets
    train_dataset = DeepFashionDataset(
        root_dir=dataset_root,
        split='train',
        transform=train_transforms,
        load_metadata=True
    )
    
    val_dataset = DeepFashionDataset(
        root_dir=dataset_root,
        split='val',
        transform=val_transforms,
        load_metadata=True
    )
    
    # Create data loaders
    if balance_classes:
        # Weighted sampling to handle class imbalance
        labels = torch.tensor(train_dataset.labels)
        class_sample_count = torch.bincount(labels, minlength=len(train_dataset.classes)).float()
        class_weights = 1.0 / torch.clamp(class_sample_count, min=1.0)
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(weights=sample_weights.double(), num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=False
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


def benchmark_dataloader(
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 100,
    warmup_batches: int = 10
) -> Dict[str, float]:
    """
    Benchmark data loader performance.
    
    Measures how fast the data loader can process batches and move data to GPU.
    This helps identify bottlenecks in the data loading pipeline.
    
    Args:
        dataloader: DataLoader to benchmark
        device: Device to move data to (CPU or GPU)
        max_batches: Maximum number of batches to process for timing
        warmup_batches: Number of warmup batches to stabilize performance
    
    Returns:
        Dictionary with performance metrics (images/sec, batches/sec, etc.)
    """
    
    print(f"Benchmarking dataloader with {max_batches} batches...")
    
    # Warmup
    print("Warming up...")
    for i, (images, labels) in enumerate(dataloader):
        if i >= warmup_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
    
    # Benchmark
    print("Running benchmark...")
    start_time = time.time()
    total_images = 0
    
    for i, (images, labels) in enumerate(dataloader):
        if i >= max_batches:
            break
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        total_images += images.size(0)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    images_per_second = total_images / total_time
    batches_per_second = max_batches / total_time
    time_per_batch = total_time / max_batches
    
    metrics = {
        'total_time': total_time,
        'total_images': total_images,
        'images_per_second': images_per_second,
        'batches_per_second': batches_per_second,
        'time_per_batch': time_per_batch,
        'batch_size': dataloader.batch_size,
        'num_workers': dataloader.num_workers
    }
    
    print(f"Benchmark Results:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Images processed: {total_images}")
    print(f"  Images/second: {images_per_second:.1f}")
    print(f"  Batches/second: {batches_per_second:.2f}")
    print(f"  Time per batch: {time_per_batch:.3f}s")
    
    return metrics


if __name__ == "__main__":
    # Example usage
    dataset_root = "./newdataset"  # Update this path
    
    # Create data loaders
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
        dataset_root=dataset_root,
        batch_size=32,
        num_workers=4,
        input_size=224,
        augmentation_strength='medium'
    )
    
    # Print dataset summaries
    train_dataset.print_summary()
    val_dataset.print_summary()
    
    # Benchmark data loaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print("\n=== Training DataLoader Benchmark ===")
    train_metrics = benchmark_dataloader(train_loader, device, max_batches=50)
    
    print("\n=== Validation DataLoader Benchmark ===")
    val_metrics = benchmark_dataloader(val_loader, device, max_batches=50)
