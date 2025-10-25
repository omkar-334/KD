# Deep Mutual Learning (DML) Knowledge Distillation

This implementation provides modular functions for Deep Mutual Learning (DML) knowledge distillation that can be easily imported and used in Jupyter notebooks or other Python scripts. DML trains multiple models simultaneously, where each model learns from the others through mutual knowledge distillation.

## Features

- **Modular Design**: Import and use individual functions in Jupyter notebooks
- **Multi-Model Training**: Train multiple models simultaneously with mutual learning
- **ResNet Support**: Support for ResNet20, ResNet32, ResNet44, ResNet56, ResNet110
- **Comprehensive Logging**: TensorBoard integration and detailed logging
- **Model Checkpointing**: Automatic saving of best models and regular checkpoints
- **Ensemble Evaluation**: Evaluate individual models and ensemble performance
- **Data Augmentation**: Optional data augmentation for improved performance
- **Resume Training**: Ability to resume training from checkpoints

## Quick Start

### Basic Usage in Jupyter

```python
import sys
sys.path.append('/Users/omkarkabde/Desktop/KD')
from DML.main import train_dml, create_dml_config, load_dml_models, evaluate_dml_models

# Create custom configuration
config = create_dml_config(
    model='ResNet32',
    data_dir='./data',
    epochs=200,
    batch_size=32,
    model_num=2,  # Number of models for DML
    lr=0.1
)

# Train the models
results = train_dml(config)

# Access results
models = results['models']
best_accuracies = results['best_accuracies']
avg_best_accuracy = results['avg_best_accuracy']
training_history = results['training_history']
```

### Using with Custom Data Loaders

```python
from model import prepare_dataloaders
from DML.main import train_dml, create_dml_config

# Prepare your own data loaders
train_loader, val_loader, test_loader = prepare_dataloaders(
    'data', batch_size=32, augment=True
)

# Create configuration
config = create_dml_config(
    model='ResNet56',
    model_num=3,  # Train 3 models
    epochs=200,
    lr=0.1
)

# Train with custom data loaders
results = train_dml(config, train_loader, val_loader, test_loader)
```

## Available Functions

### Core Training Functions

#### `train_dml(config=None, train_loader=None, val_loader=None, test_loader=None)`
Main training function that handles the complete DML training process.

**Parameters:**
- `config` (dict): Training configuration. If None, uses DEFAULT_CONFIG
- `train_loader`: Training data loader (optional if config['data_dir'] is provided)
- `val_loader`: Validation data loader (optional if config['data_dir'] is provided)
- `test_loader`: Test data loader (optional)

**Returns:**
- Dictionary containing:
  - `models`: List of trained models
  - `best_accuracies`: Best validation accuracy for each model
  - `avg_best_accuracy`: Average best accuracy across all models
  - `training_history`: List of epoch results
  - `test_results`: Test set results (if test_loader provided)
  - `config`: Final configuration used
  - `experiment_name`: Name of the experiment

#### `create_dml_config(**kwargs)`
Create a DML configuration dictionary with custom parameters.

**Parameters:**
- `**kwargs`: Configuration parameters to override defaults

**Returns:**
- Dictionary with configuration parameters

#### `load_dml_models(checkpoint_paths, device='auto')`
Load DML models from checkpoints.

**Parameters:**
- `checkpoint_paths`: List of checkpoint file paths
- `device` (str): Device to load models on

**Returns:**
- Dictionary containing:
  - `models`: List of loaded models
  - `configs`: List of model configurations
  - `device`: Device used

#### `evaluate_dml_models(models, data_loader, device='auto', config=None)`
Evaluate DML models on a dataset.

**Parameters:**
- `models`: List of trained models
- `data_loader`: Data loader for evaluation
- `device` (str): Device to use
- `config` (dict): Model configuration

**Returns:**
- List of evaluation results for each model plus ensemble results

### Utility Functions

#### `evaluate_ensemble(models, data_loader, device='auto', config=None)`
Evaluate ensemble of DML models.

#### `compute_dml_loss(models, x, y, model_idx)`
Compute DML loss for a specific model.

#### `train_epoch_dml(models, optimizers, train_loader, device, config)`
Train DML models for one epoch.

#### `validate_dml(models, val_loader, device, config)`
Validate DML models.

## Configuration Parameters

### Default Configuration
```python
DEFAULT_CONFIG = {
    'model': 'ResNet32',              # Model architecture
    'data_dir': 'data',               # Dataset directory
    'batch_size': 32,                 # Batch size
    'num_workers': 4,                 # Data loading workers
    'augment': False,                 # Data augmentation
    'epochs': 200,                    # Training epochs
    'lr': 0.1,                        # Learning rate
    'momentum': 0.9,                  # SGD momentum
    'weight_decay': 5e-4,             # Weight decay
    'step_size': 60,                  # LR scheduler step size
    'gamma': 0.1,                     # LR scheduler gamma
    'model_num': 2,                   # Number of models for DML
    'device': 'auto',                 # Device (auto/cuda/cpu)
    'print_freq': 10,                 # Print frequency
    'save_freq': 10,                  # Save frequency
    'save_dir': 'checkpoints',        # Checkpoint directory
    'log_dir': 'logs',                # Log directory
    'experiment_name': None,          # Experiment name
    'resume': None                    # Resume checkpoint
}
```

## Example Usage Patterns

### 1. Basic DML Training
```python
from DML.main import train_dml, create_dml_config

# Train 2 ResNet32 models with DML
config = create_dml_config(
    model='ResNet32',
    data_dir='./data',
    epochs=200,
    model_num=2,
    lr=0.1
)

results = train_dml(config)
print(f"Average best accuracy: {results['avg_best_accuracy']:.4f}")
```

### 2. Multi-Model DML Training
```python
from DML.main import train_dml, create_dml_config

# Train 3 ResNet56 models with DML
config = create_dml_config(
    model='ResNet56',
    data_dir='./data',
    epochs=200,
    model_num=3,
    lr=0.1,
    batch_size=64
)

results = train_dml(config)
print(f"Individual accuracies: {results['best_accuracies']}")
print(f"Average accuracy: {results['avg_best_accuracy']:.4f}")
```

### 3. Load and Evaluate Models
```python
from DML.main import load_dml_models, evaluate_dml_models

# Load trained models
checkpoint_paths = [
    'checkpoints/experiment_model1_best.pth',
    'checkpoints/experiment_model2_best.pth'
]
models_data = load_dml_models(checkpoint_paths)
models = models_data['models']

# Evaluate individual models and ensemble
test_results = evaluate_dml_models(models, test_loader)
for result in test_results:
    print(f"Model {result['model_idx']}: {result['accuracy']:.4f}")
```

### 4. Custom Model Architectures
```python
from DML.main import train_dml, create_dml_config

# Train with different ResNet architectures
config = create_dml_config(
    model='ResNet110',  # Deeper network
    data_dir='./data',
    epochs=300,
    model_num=2,
    lr=0.1,
    step_size=100,  # Adjust LR schedule for longer training
    gamma=0.1
)

results = train_dml(config)
```

### 5. Training with Custom Data
```python
from model import prepare_dataloaders
from DML.main import train_dml, create_dml_config

# Prepare data
train_loader, val_loader, test_loader = prepare_dataloaders(
    'data', batch_size=32, augment=True
)

# Train
config = create_dml_config(
    model='ResNet32',
    model_num=2,
    epochs=200
)
results = train_dml(config, train_loader, val_loader, test_loader)
```

## Model Architectures

### ResNet Variants
- **ResNet20**: 20 layers, ~0.27M parameters
- **ResNet32**: 32 layers, ~0.46M parameters (default)
- **ResNet44**: 44 layers, ~0.66M parameters
- **ResNet56**: 56 layers, ~0.85M parameters
- **ResNet110**: 110 layers, ~1.7M parameters

All models are designed for CIFAR-10/100 datasets and use the same basic ResNet architecture with different depths.

## Deep Mutual Learning Details

The DML method implements:

1. **Multiple Models**: Train multiple identical models simultaneously
2. **Mutual Knowledge Distillation**: Each model learns from all other models
3. **KL Divergence Loss**: Use KL divergence to transfer knowledge between models
4. **Ensemble Evaluation**: Evaluate both individual models and ensemble performance

### Loss Function

For each model i, the total loss is:
```
Total Loss_i = CE Loss_i + (1/(N-1)) * Σ KL(P_i || P_j) for j ≠ i
```

Where:
- **CE Loss_i**: Cross-entropy loss for model i
- **KL(P_i || P_j)**: KL divergence between model i and model j predictions
- **N**: Number of models in DML

### Training Process

1. **Forward Pass**: Each model processes the same input batch
2. **Loss Computation**: Each model computes its loss including mutual learning
3. **Backward Pass**: Each model updates its parameters independently
4. **Knowledge Transfer**: Models learn from each other's predictions

## Monitoring Training

Use TensorBoard to monitor training:

```python
# TensorBoard logs are automatically saved
# View with: tensorboard --logdir logs/
```

Available metrics:
- Training/Validation loss and accuracy for each model
- Learning rate schedule
- Ensemble performance
- Individual model performance comparison

## Output Structure

The training process creates the following structure:

```
checkpoints/
├── experiment_name_model1_best.pth      # Best model 1 checkpoint
├── experiment_name_model1_final.pth     # Final model 1 checkpoint
├── experiment_name_model2_best.pth      # Best model 2 checkpoint
├── experiment_name_model2_final.pth     # Final model 2 checkpoint
└── experiment_name_modelX_epoch_Y.pth   # Regular checkpoints

logs/
├── experiment_name.log                  # Training log
├── experiment_name_summary.json         # Experiment summary
└── experiment_name/                     # TensorBoard logs
    ├── events.out.tfevents.*
    └── ...
```

## Requirements

- PyTorch >= 1.8.0
- torchvision
- tensorboard
- tqdm
- numpy

## Notes

- The dataset should be in ImageFolder format
- Images will be automatically resized to 32x32 for CIFAR datasets
- Data augmentation includes random crops, flips, rotations, and color jittering
- Models are automatically saved as the best performing checkpoints
- Training can be resumed from any checkpoint
- All functions are designed to work seamlessly in Jupyter notebooks
- DML works best with 2-4 models for optimal performance
- Ensemble evaluation is automatically included when multiple models are used
