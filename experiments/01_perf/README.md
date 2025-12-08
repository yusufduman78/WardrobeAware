# Performance Profiling Experiments

This directory contains comprehensive performance profiling tools and experiments for the fashion recommendation system.

## Overview

The performance profiling system provides detailed analysis of:
- **Data Loading Performance**: Speed and efficiency of data pipeline
- **Model Inference Speed**: How fast models can process images
- **Memory Usage**: RAM and GPU memory consumption
- **Training Performance**: Speed and accuracy during training
- **GPU Utilization**: Hardware usage efficiency

## Directory Structure

```
experiments/01_perf/
├── README.md                 # This file
├── config.yml               # Configuration file
├── run_profile.py           # Main profiling script
├── results/                 # Results directory
│   ├── complete_profile_*.json
│   ├── benchmark_results_*.json
│   └── benchmark_summary_*.csv
└── plots/                   # Generated plots (if enabled)
```

## Quick Start

### 1. Prerequisites

Make sure you have the required dependencies:
```bash
pip install torch torchvision tqdm pyyaml psutil GPUtil matplotlib seaborn
```

### 2. Prepare Your Dataset

Ensure your dataset is organized as:
```
newdataset/
├── train/
│   ├── 1/  # category 1
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── 2/  # category 2
└── val/
    ├── 1/
    └── 2/
```

### 3. Run Performance Profile

```bash
python run_profile.py --dataset-root ./newdataset --config config.yml
```

### 4. View Results

Results will be saved in the `results/` directory with timestamps. Check the JSON files for detailed metrics and CSV files for summary data.

## Configuration

Edit `config.yml` to customize your profiling:

### Model Settings
- `model_name`: Choose from resnet18, resnet50, efficientnet_b0
- `input_size`: Image input size (default: 224)
- `num_classes`: Number of output classes (auto-detected from dataset)

### Data Loading Settings
- `batch_size`: Batch size for data loaders
- `num_workers`: Number of worker processes
- `pin_memory`: Enable memory pinning for faster GPU transfer
- `augmentation_strength`: Data augmentation level (light/medium/heavy)

### Profiling Settings
- `profile_batches`: Number of batches to profile
- `warmup_batches`: Warmup batches before timing
- `training_epochs`: Number of epochs for training benchmark

## Understanding Results

### Data Loading Metrics
- `images_per_second`: How many images can be loaded per second
- `batches_per_second`: How many batches can be processed per second
- `time_per_batch`: Average time per batch in seconds

### Model Performance Metrics
- `images_per_second`: Inference speed in images per second
- `peak_memory_usage`: Maximum memory usage during inference
- `avg_epoch_time`: Average time per training epoch
- `final_val_accuracy`: Final validation accuracy

### GPU Metrics
- `avg_gpu_utilization`: Average GPU utilization percentage
- `gpu_memory_usage`: GPU memory consumption

## Example Results

### ResNet-18 on DeepFashion (Estimated)
Based on CIFAR-10 benchmarks scaled to DeepFashion:

| Metric | Value |
|--------|-------|
| Images per second (inference) | ~2,000 |
| Epoch time (fc-only) | ~6:00 |
| Epoch time (full fine-tune) | ~9:48 - 11:28 |
| Peak memory usage | ~2-4 GB |
| GPU utilization | 42-85% |

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `batch_size` in config
   - Enable `mixed_precision: true`
   - Reduce `gpu_memory_fraction`

2. **Slow data loading**
   - Increase `num_workers`
   - Enable `pin_memory: true`
   - Enable `persistent_workers: true`

3. **Dataset not found**
   - Check dataset path in `--dataset-root`
   - Ensure proper directory structure
   - Verify image file extensions (.jpg, .png, etc.)

### Performance Tips

1. **For faster training**:
   - Use mixed precision (`mixed_precision: true`)
   - Optimize data loading (more workers, pin memory)
   - Use larger batch sizes if memory allows

2. **For accurate profiling**:
   - Use more warmup batches
   - Profile more batches for stable results
   - Run multiple times and average results

## Advanced Usage

### Custom Models

To profile custom models, modify the `create_sample_model` function in `run_profile.py`:

```python
def create_sample_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name.lower() == 'my_custom_model':
        # Your custom model implementation
        return MyCustomModel(num_classes)
    # ... existing code
```

### Custom Metrics

Add custom metrics by extending the `ModelBenchmark` class:

```python
class CustomBenchmark(ModelBenchmark):
    def custom_metric(self, dataloader):
        # Your custom metric implementation
        pass
```

## Integration with Main Project

This profiling system integrates with the main project structure:

- Uses `src/data_loader.py` for data loading
- Uses `src/benchmark.py` for model benchmarking
- Results can be used to inform model selection and optimization

## Contributing

When adding new profiling features:

1. Follow the existing code structure
2. Add comprehensive documentation
3. Include example configurations
4. Test with different model sizes and datasets
5. Update this README with new features

## References

- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [DeepFashion Dataset](https://github.com/switchablenorms/DeepFashion)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
