"""
Model Performance Benchmarking Script for Fashion Recommendation System

This script provides comprehensive benchmarking capabilities for model performance,
including speed, GPU usage, memory consumption, and accuracy metrics.

Author: isobed18
Date: 2025-10-04
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import psutil
import GPUtil
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_loader import DeepFashionDataset, create_data_loaders, benchmark_dataloader


class ModelBenchmark:
    """
    Comprehensive model benchmarking class for performance evaluation.
    
    This class provides tools to measure various aspects of model performance
    including inference speed, memory usage, training performance, and GPU utilization.
    It's designed to help optimize models and understand their computational requirements.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        results_dir: str = "./experiments/benchmark_results"
    ):
        """
        Initialize the benchmark.
        
        Sets up the benchmarking environment and prepares for performance measurement.
        
        Args:
            model: PyTorch model to benchmark
            device: Device to run on (CPU or GPU)
            results_dir: Directory to save benchmark results and logs
        """
        self.model = model
        self.device = device
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize metrics storage
        self.metrics = {
            'timestamp': datetime.now().isoformat(),
            'device': str(device),
            'model_name': model.__class__.__name__,
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    
    def benchmark_inference_speed(
        self,
        dataloader: DataLoader,
        num_batches: int = 100,
        warmup_batches: int = 10,
        mixed_precision: bool = False
    ) -> Dict[str, float]:
        """
        Benchmark model inference speed.
        
        Args:
            dataloader: DataLoader for inference
            num_batches: Number of batches to process
            warmup_batches: Number of warmup batches
            mixed_precision: Whether to use mixed precision
        
        Returns:
            Dictionary with speed metrics
        """
        print(f"Benchmarking inference speed with {num_batches} batches...")
        
        self.model.eval()
        scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= warmup_batches:
                    break
                images = images.to(self.device, non_blocking=True)
                
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        _ = self.model(images)
                else:
                    _ = self.model(images)
        
        # Benchmark
        print("Running inference benchmark...")
        start_time = time.time()
        total_images = 0
        
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                images = images.to(self.device, non_blocking=True)
                total_images += images.size(0)
                
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        _ = self.model(images)
                else:
                    _ = self.model(images)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        images_per_second = total_images / total_time
        batches_per_second = num_batches / total_time
        time_per_batch = total_time / num_batches
        time_per_image = total_time / total_images
        
        speed_metrics = {
            'total_time': total_time,
            'total_images': total_images,
            'images_per_second': images_per_second,
            'batches_per_second': batches_per_second,
            'time_per_batch': time_per_batch,
            'time_per_image': time_per_image,
            'mixed_precision': mixed_precision
        }
        
        print(f"Inference Speed Results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Images processed: {total_images}")
        print(f"  Images/second: {images_per_second:.1f}")
        print(f"  Batches/second: {batches_per_second:.2f}")
        print(f"  Time per batch: {time_per_batch:.3f}s")
        print(f"  Time per image: {time_per_image:.6f}s")
        
        return speed_metrics
    
    def benchmark_training_speed(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 1,
        learning_rate: float = 0.001,
        mixed_precision: bool = False
    ) -> Dict[str, Any]:
        """
        Benchmark model training speed and performance.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            learning_rate: Learning rate for optimizer
            mixed_precision: Whether to use mixed precision
        
        Returns:
            Dictionary with training metrics
        """
        print(f"Benchmarking training speed for {num_epochs} epochs...")
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        
        # Initialize metrics
        training_metrics = {
            'epoch_times': [],
            'train_losses': [],
            'val_losses': [],
            'train_accuracies': [],
            'val_accuracies': [],
            'gpu_utilization': [],
            'memory_usage': [],
            'mixed_precision': mixed_precision
        }
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            epoch_start = time.time()
            
            # Training phase
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, optimizer, scaler, mixed_precision
            )
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader, criterion, mixed_precision)
            
            epoch_time = time.time() - epoch_start
            
            # Store metrics
            training_metrics['epoch_times'].append(epoch_time)
            training_metrics['train_losses'].append(train_loss)
            training_metrics['val_losses'].append(val_loss)
            training_metrics['train_accuracies'].append(train_acc)
            training_metrics['val_accuracies'].append(val_acc)
            
            # GPU and memory metrics
            if torch.cuda.is_available():
                gpu_util = self._get_gpu_utilization()
                training_metrics['gpu_utilization'].append(gpu_util)
            
            memory_usage = self._get_memory_usage()
            training_metrics['memory_usage'].append(memory_usage)
            
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Calculate summary metrics
        avg_epoch_time = np.mean(training_metrics['epoch_times'])
        total_training_time = sum(training_metrics['epoch_times'])
        avg_gpu_util = np.mean(training_metrics['gpu_utilization']) if training_metrics['gpu_utilization'] else 0
        avg_memory_usage = np.mean(training_metrics['memory_usage'])
        
        training_metrics.update({
            'avg_epoch_time': avg_epoch_time,
            'total_training_time': total_training_time,
            'avg_gpu_utilization': avg_gpu_util,
            'avg_memory_usage': avg_memory_usage,
            'final_train_accuracy': training_metrics['train_accuracies'][-1],
            'final_val_accuracy': training_metrics['val_accuracies'][-1]
        })
        
        print(f"\nTraining Summary:")
        print(f"  Average epoch time: {avg_epoch_time:.2f}s")
        print(f"  Total training time: {total_training_time:.2f}s")
        print(f"  Average GPU utilization: {avg_gpu_util:.1f}%")
        print(f"  Average memory usage: {avg_memory_usage:.1f}MB")
        print(f"  Final train accuracy: {training_metrics['final_train_accuracy']:.4f}")
        print(f"  Final val accuracy: {training_metrics['final_val_accuracy']:.4f}")
        
        return training_metrics
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scaler: Optional[torch.cuda.amp.GradScaler],
        mixed_precision: bool
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if mixed_precision and scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        mixed_precision: bool
    ) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].load * 100
        except:
            pass
        return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def benchmark_memory_usage(
        self,
        dataloader: DataLoader,
        num_batches: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark memory usage during inference.
        
        Args:
            dataloader: DataLoader for inference
            num_batches: Number of batches to process
        
        Returns:
            Dictionary with memory metrics
        """
        print(f"Benchmarking memory usage with {num_batches} batches...")
        
        self.model.eval()
        
        # Measure baseline memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        baseline_memory = self._get_memory_usage()
        
        memory_usage = []
        
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                images = images.to(self.device, non_blocking=True)
                _ = self.model(images)
                
                current_memory = self._get_memory_usage()
                memory_usage.append(current_memory)
        
        # Calculate metrics
        max_memory = max(memory_usage)
        avg_memory = np.mean(memory_usage)
        memory_increase = max_memory - baseline_memory
        
        memory_metrics = {
            'baseline_memory': baseline_memory,
            'max_memory': max_memory,
            'avg_memory': avg_memory,
            'memory_increase': memory_increase,
            'peak_memory_usage': max_memory
        }
        
        print(f"Memory Usage Results:")
        print(f"  Baseline memory: {baseline_memory:.1f}MB")
        print(f"  Peak memory: {max_memory:.1f}MB")
        print(f"  Average memory: {avg_memory:.1f}MB")
        print(f"  Memory increase: {memory_increase:.1f}MB")
        
        return memory_metrics
    
    def run_comprehensive_benchmark(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a comprehensive benchmark suite.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Benchmark configuration
        
        Returns:
            Complete benchmark results
        """
        print("Starting comprehensive benchmark...")
        
        # Update base metrics
        self.metrics.update(config)
        
        # Inference speed benchmark
        print("\n" + "="*50)
        print("INFERENCE SPEED BENCHMARK")
        print("="*50)
        inference_metrics = self.benchmark_inference_speed(
            val_loader,
            num_batches=config.get('inference_batches', 100),
            mixed_precision=config.get('mixed_precision', False)
        )
        
        # Memory usage benchmark
        print("\n" + "="*50)
        print("MEMORY USAGE BENCHMARK")
        print("="*50)
        memory_metrics = self.benchmark_memory_usage(
            val_loader,
            num_batches=config.get('memory_batches', 10)
        )
        
        # Training speed benchmark
        print("\n" + "="*50)
        print("TRAINING SPEED BENCHMARK")
        print("="*50)
        training_metrics = self.benchmark_training_speed(
            train_loader,
            val_loader,
            num_epochs=config.get('training_epochs', 1),
            learning_rate=config.get('learning_rate', 0.001),
            mixed_precision=config.get('mixed_precision', False)
        )
        
        # Combine all metrics
        complete_results = {
            **self.metrics,
            'inference': inference_metrics,
            'memory': memory_metrics,
            'training': training_metrics
        }
        
        # Save results
        self.save_results(complete_results)
        
        return complete_results
    
    def save_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_path = self.results_dir / f"benchmark_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save CSV summary
        csv_path = self.results_dir / f"benchmark_summary_{timestamp}.csv"
        self._save_csv_summary(results, csv_path)
        
        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
    
    def _save_csv_summary(self, results: Dict[str, Any], csv_path: Path) -> None:
        """Save a CSV summary of the results."""
        summary_data = {
            'timestamp': results['timestamp'],
            'model_name': results['model_name'],
            'device': results['device'],
            'model_parameters': results['model_parameters'],
            'trainable_parameters': results['trainable_parameters'],
            'images_per_second': results['inference']['images_per_second'],
            'time_per_image': results['inference']['time_per_image'],
            'peak_memory_mb': results['memory']['peak_memory_usage'],
            'avg_epoch_time': results['training']['avg_epoch_time'],
            'final_train_accuracy': results['training']['final_train_accuracy'],
            'final_val_accuracy': results['training']['final_val_accuracy'],
            'avg_gpu_utilization': results['training']['avg_gpu_utilization']
        }
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_data.keys())
            writer.writeheader()
            writer.writerow(summary_data)


def create_benchmark_config(
    model_name: str,
    batch_size: int = 32,
    input_size: int = 224,
    mixed_precision: bool = False,
    inference_batches: int = 100,
    memory_batches: int = 10,
    training_epochs: int = 1,
    learning_rate: float = 0.001
) -> Dict[str, Any]:
    """
    Create a benchmark configuration.
    
    Args:
        model_name: Name of the model being benchmarked
        batch_size: Batch size for data loaders
        input_size: Input image size
        mixed_precision: Whether to use mixed precision
        inference_batches: Number of batches for inference benchmark
        memory_batches: Number of batches for memory benchmark
        training_epochs: Number of epochs for training benchmark
        learning_rate: Learning rate for training
    
    Returns:
        Benchmark configuration dictionary
    """
    return {
        'model_name': model_name,
        'batch_size': batch_size,
        'input_size': input_size,
        'mixed_precision': mixed_precision,
        'inference_batches': inference_batches,
        'memory_batches': memory_batches,
        'training_epochs': training_epochs,
 'learning_rate': learning_rate
    }


if __name__ == "__main__":
    # Example usage
    from torchvision import models
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a sample model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)  # 10 classes for example
    
    # Create benchmark
    benchmark = ModelBenchmark(model, device)
    
    # Create sample data loaders (you'll need to provide actual dataset path)
    # train_loader, val_loader, _, _ = create_data_loaders(
    #     dataset_root="./newdataset",
    #     batch_size=32,
    #     num_workers=4
    # )
    
    # Create config
    config = create_benchmark_config(
        model_name="ResNet18",
        batch_size=32,
        mixed_precision=False
    )
    
    # Run benchmark (uncomment when you have data)
    # results = benchmark.run_comprehensive_benchmark(train_loader, val_loader, config)
