"""
Performance Profiling Script for Fashion Recommendation System

This script provides comprehensive performance profiling capabilities for models,
including speed, GPU usage, memory consumption, and data loading efficiency.

Author: Development Team
Date: 2025-01-11
"""

import os
import sys
import time
import json
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from data_loader import create_data_loaders, benchmark_dataloader
from benchmark import ModelBenchmark, create_benchmark_config


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_sample_model(model_name: str, num_classes: int) -> nn.Module:
    """Create a sample model for testing."""
    if model_name.lower() == 'resnet18':
        from torchvision import models
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.lower() == 'resnet50':
        from torchvision import models
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.lower() == 'efficientnet_b0':
        from torchvision import models
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def profile_data_loading(
    dataset_root: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Profile data loading performance."""
    print("="*60)
    print("DATA LOADING PERFORMANCE PROFILE")
    print("="*60)
    
    # Create data loaders
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
        dataset_root=dataset_root,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        input_size=config['input_size'],
        augmentation_strength=config['augmentation_strength'],
        persistent_workers=config['persistent_workers']
    )
    
    # Print dataset info
    train_dataset.print_summary()
    val_dataset.print_summary()
    
    # Benchmark data loaders
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nBenchmarking with device: {device}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Number of workers: {config['num_workers']}")
    print(f"Pin memory: {config['pin_memory']}")
    print(f"Persistent workers: {config['persistent_workers']}")
    
    # Profile training data loader
    print(f"\n--- Training DataLoader Profile ---")
    train_metrics = benchmark_dataloader(
        train_loader, 
        device, 
        max_batches=config['profile_batches'],
        warmup_batches=config['warmup_batches']
    )
    
    # Profile validation data loader
    print(f"\n--- Validation DataLoader Profile ---")
    val_metrics = benchmark_dataloader(
        val_loader, 
        device, 
        max_batches=config['profile_batches'],
        warmup_batches=config['warmup_batches']
    )
    
    return {
        'train_loader': train_metrics,
        'val_loader': val_metrics,
        'dataset_info': {
            'train_images': len(train_dataset),
            'val_images': len(val_dataset),
            'num_classes': len(train_dataset.classes),
            'classes': train_dataset.classes
        }
    }


def profile_model_performance(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Profile model performance."""
    print("="*60)
    print("MODEL PERFORMANCE PROFILE")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create benchmark
    benchmark = ModelBenchmark(model, device)
    
    # Create benchmark config
    benchmark_config = create_benchmark_config(
        model_name=config['model_name'],
        batch_size=config['batch_size'],
        input_size=config['input_size'],
        mixed_precision=config['mixed_precision'],
        inference_batches=config['inference_batches'],
        memory_batches=config['memory_batches'],
        training_epochs=config['training_epochs'],
        learning_rate=config['learning_rate']
    )
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(
        train_loader, val_loader, benchmark_config
    )
    
    return results


def run_complete_profile(
    dataset_root: str,
    config_path: str,
    output_dir: str = "./experiments/01_perf/results"
) -> Dict[str, Any]:
    """Run complete performance profile."""
    print("Starting complete performance profile...")
    print(f"Dataset root: {dataset_root}")
    print(f"Config path: {config_path}")
    print(f"Output directory: {output_dir}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Profile data loading
    data_profile = profile_data_loading(dataset_root, config)
    
    # Create model
    print(f"\nCreating model: {config['model_name']}")
    model = create_sample_model(
        config['model_name'], 
        data_profile['dataset_info']['num_classes']
    )
    
    # Create data loaders for model profiling
    train_loader, val_loader, _, _ = create_data_loaders(
        dataset_root=dataset_root,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        input_size=config['input_size'],
        augmentation_strength=config['augmentation_strength'],
        persistent_workers=config['persistent_workers']
    )
    
    # Profile model performance
    model_profile = profile_model_performance(model, train_loader, val_loader, config)
    
    # Combine results
    complete_results = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'data_profile': data_profile,
        'model_profile': model_profile
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"complete_profile_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(complete_results, f, indent=2)
    
    print(f"\nComplete profile results saved to: {results_file}")
    
    # Print summary
    print_summary(complete_results)
    
    return complete_results


def print_summary(results: Dict[str, Any]) -> None:
    """Print a summary of the results."""
    print("\n" + "="*60)
    print("PERFORMANCE PROFILE SUMMARY")
    print("="*60)
    
    # Data loading summary
    data_profile = results['data_profile']
    print(f"\nData Loading Performance:")
    print(f"  Training images/sec: {data_profile['train_loader']['images_per_second']:.1f}")
    print(f"  Validation images/sec: {data_profile['val_loader']['images_per_second']:.1f}")
    print(f"  Dataset size: {data_profile['dataset_info']['train_images']} train, {data_profile['dataset_info']['val_images']} val")
    print(f"  Number of classes: {data_profile['dataset_info']['num_classes']}")
    
    # Model performance summary
    model_profile = results['model_profile']
    print(f"\nModel Performance:")
    print(f"  Model: {model_profile['model_name']}")
    print(f"  Parameters: {model_profile['model_parameters']:,}")
    print(f"  Trainable parameters: {model_profile['trainable_parameters']:,}")
    print(f"  Inference speed: {model_profile['inference']['images_per_second']:.1f} images/sec")
    print(f"  Peak memory usage: {model_profile['memory']['peak_memory_usage']:.1f} MB")
    print(f"  Average epoch time: {model_profile['training']['avg_epoch_time']:.2f} seconds")
    print(f"  Final validation accuracy: {model_profile['training']['final_val_accuracy']:.4f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run performance profile')
    parser.add_argument('--dataset-root', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='./experiments/01_perf/results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset_root):
        print(f"Error: Dataset root not found: {args.dataset_root}")
        sys.exit(1)
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Run profile
    try:
        results = run_complete_profile(
            dataset_root=args.dataset_root,
            config_path=args.config,
            output_dir=args.output_dir
        )
        print("\nPerformance profile completed successfully!")
        
    except Exception as e:
        print(f"Error during profiling: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
