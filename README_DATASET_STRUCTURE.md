# Knowledge Distillation with Dataset-Specific Organization

This repository now supports running experiments on multiple datasets with organized output structure.

## 📁 New Directory Structure

```
checkpoints/{dataset_name}/
├── teachers/
│   ├── ResNet50_teacher_best.pth
│   └── ResNet50_teacher_final.pth
├── AT/
│   ├── AT_ResNet18_Final_*.pth
│   └── ...
├── BYOT/
│   ├── BYOT_ResNet18_Final_*.pth
│   └── ...
├── DML/
│   ├── DML_ResNet18_Final_*.pth
│   └── ...
└── FitNets/
    ├── FitNets_ResNet18_Final_*.pth
    └── ...

logs/{dataset_name}/
├── teachers/
│   └── ResNet50_Teacher_Final_*/
│       ├── ResNet50_Teacher_Final_*.log
│       └── events.out.tfevents.*
├── AT/
│   └── AT_ResNet18_Final_*/
├── BYOT/
│   └── BYOT_ResNet18_Final_*/
├── DML/
│   └── DML_ResNet18_Final_*/
└── FitNets/
    └── FitNets_ResNet18_Final_*/
```

## 🚀 Usage

### Single Dataset Experiment

```bash
# Run on a specific dataset
python run_experiment.py

# Or with custom parameters
python -c "
from run_experiment import main
main(data_dir='data/cifar10', dataset_name='cifar10', batch_size=32)
"
```

### Multiple Dataset Experiments

```bash
# Run on multiple datasets
python run_multi_dataset.py
```

### Direct Script Usage

```python
from final import main

# Run experiment on specific dataset
comparison_df, summary = main(
    data_dir="data/cifar10",
    dataset_name="cifar10", 
    batch_size=32
)
```

## 📊 Dataset Configuration

### Example: CIFAR-10
```
data/cifar10/
├── train/
│   ├── airplane/
│   ├── automobile/
│   ├── bird/
│   └── ...
└── val/
    ├── airplane/
    ├── automobile/
    ├── bird/
    └── ...
```

### Example: CIFAR-100
```
data/cifar100/
├── train/
│   ├── apple/
│   ├── aquarium_fish/
│   ├── baby/
│   └── ...
└── val/
    ├── apple/
    ├── aquarium_fish/
    ├── baby/
    └── ...
```

## 🔧 Configuration Examples

### Custom Dataset Setup

```python
# In run_multi_dataset.py, add your dataset:
datasets = [
    {
        "name": "my_custom_dataset",
        "data_dir": "data/my_custom_dataset",
        "batch_size": 32
    },
    {
        "name": "large_images",
        "data_dir": "data/large_images", 
        "batch_size": 16  # Smaller batch for large images
    }
]
```

### Individual Method Training

```python
from AT.main import train_at, create_at_config
from teacher import train_teacher, create_teacher_config

# 1. Train teacher for specific dataset
teacher_config = create_teacher_config(
    model="ResNet50",
    data_dir="data/cifar10",
    epochs=100
)
teacher_results = train_teacher(teacher_config, dataset_name="cifar10")

# 2. Train student with teacher
at_config = create_at_config(
    model="ResNet18",
    data_dir="data/cifar10",
    epochs=50
)
at_results = train_at(at_config, dataset_name="cifar10")
```

## 📈 Output Organization

### Per-Dataset Results

Each dataset gets its own organized output:

```
logs/final_experiment_20241201_143022/
├── final_experiment.log
├── comparison_results.json
├── comparison_results.csv
└── experiment_summary.json

checkpoints/cifar10/teachers/
├── ResNet50_teacher_best.pth
└── ResNet50_teacher_final.pth

logs/cifar10/teachers/
└── ResNet50_Teacher_Final_20241201_143022/
    ├── ResNet50_Teacher_Final_20241201_143022.log
    └── events.out.tfevents.*
```

### Multi-Dataset Summary

```
logs/multi_dataset_results_20241201_143022.json
```

## 🎯 Benefits of Dataset-Specific Organization

1. **Isolation**: Each dataset's experiments are completely separate
2. **Scalability**: Easy to add new datasets without conflicts
3. **Comparison**: Easy to compare results across datasets
4. **Storage**: Efficient organization of checkpoints and logs
5. **Reproducibility**: Clear mapping between datasets and results

## 🔄 Migration from Old Structure

If you have existing experiments in the old structure:

```bash
# Old structure
checkpoints/teachers/
logs/teachers/

# New structure  
checkpoints/{dataset_name}/teachers/
logs/{dataset_name}/teachers/
```

The system automatically creates the new structure when you run experiments with `dataset_name` parameter.

## 🧪 Testing

```bash
# Test the setup
python test_setup.py

# Test with specific dataset
python -c "
from test_setup import test_directory_structure
test_directory_structure()
"
```

## 📝 Example Workflow

1. **Prepare datasets**:
   ```bash
   # Organize your datasets
   mkdir -p data/{cifar10,cifar100,imagenet_subset}
   # Copy your data into respective directories
   ```

2. **Run single experiment**:
   ```bash
   python run_experiment.py
   ```

3. **Run multi-dataset experiments**:
   ```bash
   python run_multi_dataset.py
   ```

4. **Analyze results**:
   ```bash
   # Check individual dataset results
   ls logs/*/comparison_results.csv
   
   # Check multi-dataset summary
   cat logs/multi_dataset_results_*.json
   ```

## 🎛️ Advanced Usage

### Custom Teacher Models

```python
# Train different teacher architectures for different datasets
teacher_configs = {
    "cifar10": create_teacher_config(model="ResNet50", epochs=100),
    "cifar100": create_teacher_config(model="ResNet50", epochs=120),  # More epochs for harder task
    "imagenet": create_teacher_config(model="ResNet101", epochs=80),  # Larger model for complex data
}
```

### Method-Specific Parameters

```python
# Different parameters for different datasets
at_configs = {
    "cifar10": create_at_config(beta=1e3, epochs=50),
    "cifar100": create_at_config(beta=2e3, epochs=60),  # More attention transfer for harder task
    "imagenet": create_at_config(beta=5e3, epochs=40),  # Different balance for large dataset
}
```

This new structure makes it easy to run comprehensive experiments across multiple datasets while keeping everything organized and reproducible! 🎉
