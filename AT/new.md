# Attention Transfer (AT) Knowledge Distillation

This implementation provides modular functions for Attention Transfer (AT) knowledge distillation that can be easily imported and used in Jupyter notebooks or other Python scripts. AT transfers attention maps from a teacher model to a student model to improve learning.

## Features

- **Modular Design**: Import and use individual functions in Jupyter notebooks
- **Teacher-Student Training**: Support for pre-trained teacher models
- **Attention Map Transfer**: Transfer attention patterns from teacher to student
- **Multi-Model Support**: MultiBranchNet, ResNet18, ResNet50
- **Comprehensive Logging**: TensorBoard integration and detailed logging
- **Model Checkpointing**: Automatic saving of best models and regular checkpoints
- **Data Augmentation**: Optional data augmentation for improved performance
- **Resume Training**: Ability to resume training from checkpoints

## Quick Start

### Basic Usage in Jupyter

```python
import sys
sys.path.append('/Users/omkarkabde/Desktop/KD')
from AT.main_new import train_at, create_at_config

# Create custom configuration
config = create_at_config(
    model='MultiBranchNet',
    data_dir='./data',
    epochs=50,
    batch_size=32,
    beta=1e3  # Attention transfer weight
)

# Train the model
results = train_at(config)

# Access results
model = results['model']
best_accuracy = results['best_accuracy']
training_history = results['training_history']
```

### Using with Teacher Model

```python
from AT.main_new import train_at, create_at_config

# Train with teacher model
config = create_at_config(
    model='ResNet18',
    teacher_model='path/to/teacher.pth',
    data_dir='./data',
    epochs=50,
    alpha=0.1,  # Enable KD
    beta=1e3    # Attention transfer
)

results = train_at(config)
```

## Available Functions

### Core Training Functions

#### `train_at(config=None, train_loader=None, val_loader=None, test_loader=None)`
Main training function that handles the complete AT training process.

**Parameters:**
- `config` (dict): Training configuration. If None, uses DEFAULT_CONFIG
- `train_loader`: Training data loader (optional if config['data_dir'] is provided)
- `val_loader`: Validation data loader (optional if config['data_dir'] is provided)
- `test_loader`: Test data loader (optional)

**Returns:**
- Dictionary containing:
  - `model`: Trained student model
  - `teacher_model`: Teacher model (if provided)
  - `best_accuracy`: Best validation accuracy
  - `training_history`: List of epoch results
  - `test_results`: Test set results (if test_loader provided)
  - `config`: Final configuration used
  - `experiment_name`: Name of the experiment

#### `create_at_config(**kwargs)`
Create an AT configuration dictionary with custom parameters.

**Parameters:**
- `**kwargs`: Configuration parameters to override defaults

**Returns:**
- Dictionary with configuration parameters

#### `visualize_attention_maps(model, x, config, layer_idx=0)`
Visualize attention maps for a given input.

**Parameters:**
- `model`: Trained model
- `x`: Input tensor
- `config`: Model configuration
- `layer_idx`: Which layer to visualize (0-based index)

**Returns:**
- `torch.Tensor`: Attention map

### Utility Functions

#### `load_teacher_model(teacher_path, num_classes, device)`
Load teacher model from checkpoint.

#### `extract_attention_maps(model, x, config)`
Extract attention maps from model features.

#### `compute_at_loss(student_model, teacher_model, x, y, config)`
Compute Attention Transfer loss.

#### `train_epoch(student_model, teacher_model, train_loader, optimizer, device, config)`
Train for one epoch.

#### `validate(student_model, teacher_model, val_loader, device, config)`
Validate the model.

## Configuration Parameters

### Default Configuration
```python
DEFAULT_CONFIG = {
    'model': 'MultiBranchNet',              # Model architecture
    'teacher_model': None,                  # Path to teacher model
    'data_dir': 'data',                     # Dataset directory
    'batch_size': 32,                       # Batch size
    'num_workers': 4,                       # Data loading workers
    'augment': False,                       # Data augmentation
    'epochs': 50,                           # Training epochs
    'lr': 0.01,                             # Learning rate
    'momentum': 0.9,                        # SGD momentum
    'weight_decay': 5e-4,                   # Weight decay
    'step_size': 10,                        # LR scheduler step size
    'gamma': 0.1,                           # LR scheduler gamma
    'temperature': 4.0,                     # KD temperature
    'alpha': 0.0,                           # Weight for KD (0 = no KD)
    'beta': 1e3,                            # Weight for attention transfer
    'device': 'auto',                       # Device (auto/cuda/cpu)
    'print_freq': 100,                      # Print frequency
    'save_freq': 10,                        # Save frequency
    'save_dir': 'checkpoints',              # Checkpoint directory
    'log_dir': 'logs',                      # Log directory
    'experiment_name': None,                # Experiment name
    'resume': None                          # Resume checkpoint
}
```

## Example Usage Patterns

### 1. Basic AT Training
```python
from AT.main_new import train_at, create_at_config

# Train MultiBranchNet with AT
config = create_at_config(
    model='MultiBranchNet',
    data_dir='./data',
    epochs=50,
    beta=1e3
)

results = train_at(config)
print(f"Best accuracy: {results['best_accuracy']:.4f}")
```

### 2. Training with Teacher Model
```python
from AT.main_new import train_at, create_at_config

# Train with teacher model
config = create_at_config(
    model='ResNet18',
    teacher_model='path/to/teacher.pth',
    data_dir='./data',
    epochs=50,
    alpha=0.1,  # Enable KD
    beta=1e3    # Attention transfer
)

results = train_at(config)
```

### 3. Custom Data Loaders
```python
from model import prepare_dataloaders
from AT.main_new import train_at, create_at_config

# Prepare custom data loaders
train_loader, val_loader, test_loader = prepare_dataloaders(
    'data', batch_size=32, augment=True
)

# Train with custom data
config = create_at_config(model='MultiBranchNet', epochs=50)
results = train_at(config, train_loader, val_loader, test_loader)
```

### 4. Load and Evaluate Model
```python
from shared_utils import load_model, evaluate_model
from AT.main_new import create_at_config

# Load trained model
checkpoint = load_model('checkpoints/model_best.pth')
model = checkpoint['model']
config = checkpoint['config']

# Evaluate
test_results = evaluate_model(model, test_loader, config=config)
print(f"Test accuracy: {test_results['accuracy']:.4f}")
```

### 5. Visualize Attention Maps
```python
from AT.main_new import visualize_attention_maps
import torch

# Load model and visualize attention
x = torch.randn(1, 3, 128, 128)
attention_map = visualize_attention_maps(model, x, config, layer_idx=0)
print(f"Attention map shape: {attention_map.shape}")
```

## Model Architectures

### Supported Models
- **MultiBranchNet**: Custom multi-branch network with ECA layers
- **ResNet18**: ResNet-18 with intermediate outputs
- **ResNet50**: ResNet-50 with intermediate outputs

All models are designed to work with your `prepare_dataloaders` function and support intermediate feature extraction for attention transfer.

## Attention Transfer Details

The AT method implements:

1. **Attention Map Extraction**: Extract attention maps from intermediate layers
2. **Attention Transfer Loss**: Transfer attention patterns from teacher to student
3. **Knowledge Distillation**: Optional KD with attention transfer
4. **Multi-Scale Attention**: Transfer attention at multiple scales

### Loss Function

The AT method uses:
```
Total Loss = Classification Loss + α × KD Loss + β × AT Loss
```

Where:
- **Classification Loss**: Standard cross-entropy loss
- **KD Loss**: Knowledge distillation loss (if teacher provided)
- **AT Loss**: Attention transfer loss between teacher and student
- **α**: Weight for knowledge distillation (0 = no KD)
- **β**: Weight for attention transfer

### Attention Map Computation

Attention maps are computed using:
```python
def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()
```

## Monitoring Training

Use TensorBoard to monitor training:

```python
# TensorBoard logs are automatically saved
# View with: tensorboard --logdir logs/
```

Available metrics:
- Training/Validation loss and accuracy
- Learning rate schedule
- Detailed loss components (classification, KD, AT)
- Attention transfer metrics

## Output Structure

The training process creates the following structure:

```
checkpoints/
├── experiment_name_best.pth              # Best model checkpoint
├── experiment_name_final.pth             # Final model checkpoint
└── experiment_name_epoch_X.pth           # Regular checkpoints

logs/
├── experiment_name.log                   # Training log
├── experiment_name_summary.json          # Experiment summary
└── experiment_name/                      # TensorBoard logs
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
- Images will be automatically resized to 128x128 for MultiBranchNet
- Data augmentation includes random crops, flips, rotations, and color jittering
- Models are automatically saved as the best performing checkpoints
- Training can be resumed from any checkpoint
- All functions are designed to work seamlessly in Jupyter notebooks
- AT works best with teacher models that have intermediate feature outputs
- Attention maps are computed from intermediate feature maps
