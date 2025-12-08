
"""
Training Pipeline for Fashion Recommendation System

This module provides a comprehensive training pipeline for training models
on the DeepFashion dataset for clothing DNA extraction.

Author: isobed18
Date: 2025-10-04
"""

import os
import sys
import time
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_loader import create_data_loaders, DeepFashionDataset, benchmark_dataloader
from benchmark import ModelBenchmark


class FashionTrainer:
    """
    Comprehensive trainer for fashion recommendation models.

    This class handles the complete training process including model setup,
    training loop, validation, checkpointing, and monitoring. It's designed
    to be flexible and support various model architectures and training strategies.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Dict[str, Any],
        save_dir: str = "./experiments/training"
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            device: Device to train on
            config: Training configuration
            save_dir: Directory to save checkpoints and results
        """
        self.model = model
        self.device = device
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Move model to device
        self.model = self.model.to(device)

        # Initialize training components
        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        self.scaler = GradScaler() if config.get('mixed_precision', False) else None

        # Initialize tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []

        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def _get_criterion(self) -> nn.Module:
        """Get loss function."""
        if self.config.get('use_class_weights', False):
            weights = self.config.get('class_weights')
            if isinstance(weights, list):
                w = torch.tensor(weights, dtype=torch.float32, device=self.device)
                return nn.CrossEntropyLoss(weight=w)
            return nn.CrossEntropyLoss()
        return nn.CrossEntropyLoss()

    def _get_optimizer(self) -> optim.Optimizer:
        """Get optimizer."""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        learning_rate = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-4)

        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _get_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler."""
        scheduler_name = self.config.get('scheduler', 'none').lower()

        if scheduler_name == 'none':
            return None
        elif scheduler_name == 'step':
            step_size = self.config.get('step_size', 30)
            gamma = self.config.get('gamma', 0.1)
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_name == 'cosine':
            T_max = self.config.get('max_epochs', 100)
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max
            )
        elif scheduler_name == 'plateau':
            patience = self.config.get('patience', 10)
            factor = self.config.get('factor', 0.5)
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=patience,
                factor=factor
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")

            for images, labels in progress_bar:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if self.scaler:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int = 100,
        early_stopping_patience: int = 20
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            max_epochs: Maximum number of epochs
            early_stopping_patience: Early stopping patience

        Returns:
            Training results dictionary
        """
        print(f"Starting training for {max_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Mixed precision: {self.scaler is not None}")

        start_time = time.time()
        epoch_throughputs: List[float] = []

        for epoch in range(max_epochs):
            epoch_start = time.time()

            # Training
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)

            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()

            # Track metrics
            epoch_time = time.time() - epoch_start
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(current_lr)
            self.epoch_times.append(epoch_time)

            # Lightweight throughput estimate (images/sec)
            try:
                imgs_per_epoch = len(train_loader) * train_loader.batch_size
                throughput = imgs_per_epoch / max(epoch_time, 1e-6)
                epoch_throughputs.append(throughput)
            except Exception:
                epoch_throughputs.append(0.0)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{max_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Epoch Time: {epoch_time:.2f}s")

            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1

            # Save regular checkpoint
            if (epoch + 1) % self.config.get('save_every', 10) == 0:
                self.save_checkpoint(epoch, is_best=False)

            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                print(f"Best validation accuracy: {self.best_val_accuracy:.4f} at epoch {self.best_epoch + 1}")
                break

        total_time = time.time() - start_time

        # Create results summary
        results = {
            'config': self.config,
            'total_time': total_time,
            'total_epochs': len(self.train_losses),
            'best_epoch': self.best_epoch,
            'best_val_accuracy': self.best_val_accuracy,
            'final_train_accuracy': self.train_accuracies[-1],
            'final_val_accuracy': self.val_accuracies[-1],
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'avg_epoch_time': np.mean(self.epoch_times),
            'epoch_throughputs': epoch_throughputs
        }

        # Save final results
        self.save_results(results)
        self.plot_training_curves()

        return results

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch + 1}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  New best model saved! Val Acc: {self.best_val_accuracy:.4f}")

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save training results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON results
        results_path = self.save_dir / f'training_results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Training results saved to: {results_path}")

    def plot_training_curves(self) -> None:
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Accuracy curves
        axes[0, 1].plot(self.train_accuracies, label='Train Acc')
        axes[0, 1].plot(self.val_accuracies, label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Learning rate curve
        axes[1, 0].plot(self.learning_rates)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)

        # Epoch time curve
        axes[1, 1].plot(self.epoch_times)
        axes[1, 1].set_title('Epoch Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)

        plt.tight_layout()

        # Save plot
        plot_path = self.save_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Training curves saved to: {plot_path}")


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Create a model for training."""
    if model_name.lower() == 'resnet18':
        from torchvision import models
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.lower() == 'resnet50':
        from torchvision import models
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.lower() == 'efficientnet_b0':
        from torchvision import models
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train fashion recommendation model')
    parser.add_argument('--dataset-root', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration file')
    parser.add_argument('--output-dir', type=str, default='./experiments/training',
                       help='Output directory for checkpoints and results')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    # Optional benchmarking hooks
    parser.add_argument('--bench-loader', type=int, default=0,
                        help='Profile data loader/device transfer for N batches (0 to disable)')
    parser.add_argument('--bench-infer', action='store_true',
                        help='Profile pure model inference on a dummy batch')
    parser.add_argument('--bench-only', action='store_true',
                        help='Run benchmarks only and exit')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, train_dataset, val_dataset = create_data_loaders(
        dataset_root=args.dataset_root,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=config['pin_memory'],
        input_size=config['input_size'],
        augmentation_strength=config['augmentation_strength'],
        persistent_workers=config['persistent_workers'],
        balance_classes=config.get('balance_classes', False)
    )

    # Print dataset info
    train_dataset.print_summary()
    val_dataset.print_summary()

    # Optionally compute and attach class weights for imbalance
    if config.get('use_class_weights', False):
        cw = train_dataset.get_class_weights().tolist()
        config['class_weights'] = cw

    # Create model
    print(f"Creating model: {config['model_name']}")
    model = create_model(
        model_name=config['model_name'],
        num_classes=len(train_dataset.classes),
        pretrained=config.get('pretrained', True)
    )

    # Create trainer
    trainer = FashionTrainer(
        model=model,
        device=device,
        config=config,
        save_dir=args.output_dir
    )

    # Optional: lightweight benchmarks before training
    if args.bench_loader or args.bench_infer:
        args_output = Path(args.output_dir)
        args_output.mkdir(parents=True, exist_ok=True)
        bench = ModelBenchmark(trainer.model, device=device, results_dir=str(args_output))
        bench_results: Dict[str, Any] = {}

        if args.bench_loader:
            print(f"\n[Benchmark] DataLoader transfer for {args.bench_loader} batches (train)...")
            train_dl_metrics = benchmark_dataloader(train_loader, device, max_batches=args.bench_loader)
            print(f"[Benchmark] DataLoader transfer for {args.bench_loader} batches (val)...")
            val_dl_metrics = benchmark_dataloader(val_loader, device, max_batches=args.bench_loader)
            bench_results['loader_train'] = train_dl_metrics
            bench_results['loader_val'] = val_dl_metrics

        if args.bench_infer:
            print("\n[Benchmark] Pure model inference speed on validation loader...")
            infer_metrics = bench.benchmark_inference_speed(
                dataloader=val_loader,
                num_batches=min(len(val_loader), max(args.bench_loader, 100)),
                warmup_batches=max(5, min(20, len(val_loader)//10)),
                mixed_precision=bool(config.get('mixed_precision', False))
            )
            bench_results['inference'] = infer_metrics

        # Save benchmark results next to training outputs
        with (args_output / 'bench_results.json').open('w') as fh:
            json.dump(bench_results, fh, indent=2)
        print(f"[Benchmark] Results saved to: {args_output / 'bench_results.json'}")

        if args.bench_only:
            return

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and trainer.scheduler:
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Train model
    results = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=config.get('max_epochs', 100),
        early_stopping_patience=config.get('early_stopping_patience', 20)
    )

    print("\nTraining completed!")
    print(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Total training time: {results['total_time']:.2f} seconds")


if __name__ == "__main__":
    main()
