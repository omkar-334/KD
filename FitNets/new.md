# FitNets Knowledge Distillation

This implementation provides modular functions for FitNets knowledge distillation that can be easily imported and used in Jupyter notebooks or other Python scripts. FitNets transfers knowledge by matching intermediate feature representations between teacher and student models.

## Features

- **Modular Design**: Import and use individual functions in Jupyter notebooks
- **Teacher-Student Training**: Support for pre-trained teacher models
- **Feature Matching**: Match intermediate feature representations
- **Knowledge Distillation**: Optional KD with feature matching
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
from FitNets.main import train_fitnets, create_fitnets_config

# Create custom configuration
config = create_fitnets_config(
    model='MultiBranchNet',
    data_dir='./data',
    epochs=50,
    batch_size=32,
    beta=1e-3  # Feature matching weight
)

# Train the model
results = train_fitnets(config)

# Access results
model = results['model']
best_accuracy = results['best_accuracy']
training_history = results['training_history']
```

### Using with Teacher Model

```python
from FitNets.main import train_fitnets, create_fitnets_config

# Train with teacher model
config = create_fitnets_config(
    model='ResNet18',
    teacher_model='path/to/teacher.pth',
    data_dir='./data',
    epochs=50,
    alpha=0.1,  # Enable KD
    beta=1e-3   # Feature matching
)

results = train_fitnets(config)
```

## Available Functions

### Core Training Functions

#### `train_fitnets(config=None, train_loader=None, val_loader=None, test_loader=None)`
Main training function that handles the complete FitNets training process.

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

#### `create_fitnets_config(**kwargs)`
Create a FitNets configuration dictionary with custom parameters.

**Parameters:**
- `**kwargs`: Configuration parameters to override defaults

**Returns:**
- Dictionary with configuration parameters

#### `visualize_feature_maps(model, x, config, layer_idx=0)`
Visualize feature maps for a given input.

**Parameters:**
- `model`: Trained model
- `x`: Input tensor
- `config`: Model configuration
- `layer_idx`: Which layer to visualize (0-based index)

**Returns:**
- `torch.Tensor`: Feature map

### Utility Functions

#### `load_teacher_model(teacher_path, num_classes, device)`
Load teacher model from checkpoint.

#### `extract_teacher_features(teacher_model, x, config)`
Extract features from teacher model for FitNets.

#### `extract_student_features(student_model, x, config)`
Extract features from student model for FitNets.

#### `compute_fitnets_loss(student_model, teacher_model, x, y, config)`
Compute FitNets loss.

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
    'temperature': 3.0,                     # KD temperature
    'alpha': 0.1,                           # Weight for KD
    'beta': 1e-3,                           # Weight for feature matching
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

### 1. Basic FitNets Training
```python
from FitNets.main import train_fitnets, create_fitnets_config

# Train MultiBranchNet with FitNets
config = create_fitnets_config(
    model='MultiBranchNet',
    data_dir='./data',
    epochs=50,
    beta=1e-3
)

results = train_fitnets(config)
print(f"Best accuracy: {results['best_accuracy']:.4f}")
```

### 2. Training with Teacher Model
```python
from FitNets.main import train_fitnets, create_fitnets_config

# Train with teacher model
config = create_fitnets_config(
    model='ResNet18',
    teacher_model='path/to/teacher.pth',
    data_dir='./data',
    epochs=50,
    alpha=0.1,  # Enable KD
    beta=1e-3   # Feature matching
)

results = train_fitnets(config)
```

### 3. Custom Data Loaders
```python
from model import prepare_dataloaders
from FitNets.main import train_fitnets, create_fitnets_config

# Prepare custom data loaders
train_loader, val_loader, test_loader = prepare_dataloaders(
    'data', batch_size=32, augment=True
)

# Train with custom data
config = create_fitnets_config(model='MultiBranchNet', epochs=50)
results = train_fitnets(config, train_loader, val_loader, test_loader)
```

### 4. Load and Evaluate Model
```python
from shared_utils import load_model, evaluate_model
from FitNets.main import create_fitnets_config

# Load trained model
checkpoint = load_model('checkpoints/model_best.pth')
model = checkpoint['model']
config = checkpoint['config']

# Evaluate
test_results = evaluate_model(model, test_loader, config=config)
print(f"Test accuracy: {test_results['accuracy']:.4f}")
```

### 5. Visualize Feature Maps
```python
from FitNets.main import visualize_feature_maps
import torch

# Load model and visualize features
x = torch.randn(1, 3, 128, 128)
feature_map = visualize_feature_maps(model, x, config, layer_idx=0)
print(f"Feature map shape: {feature_map.shape}")
```

## Model Architectures

### Supported Models
- **MultiBranchNet**: Custom multi-branch network with ECA layers
- **ResNet18**: ResNet-18 with intermediate outputs
- **ResNet50**: ResNet-50 with intermediate outputs

All models are designed to work with your `prepare_dataloaders` function and support intermediate feature extraction for FitNets training.

## FitNets Details

The FitNets method implements:

1. **Feature Matching**: Match intermediate feature representations between teacher and student
2. **Knowledge Distillation**: Optional KD with feature matching
3. **Feature Alignment**: Align student features with teacher features
4. **Multi-Scale Learning**: Learn at multiple scales simultaneously

### Loss Function

The FitNets method uses:
```
Total Loss = Classification Loss + α × KD Loss + β × Feature Matching Loss
```

Where:
- **Classification Loss**: Standard cross-entropy loss
- **KD Loss**: Knowledge distillation loss (if teacher provided)
- **Feature Loss**: MSE loss between student and teacher features
- **α**: Weight for knowledge distillation
- **β**: Weight for feature matching

### Feature Matching

Feature matching is implemented as:
```python
def extract_teacher_features(teacher_model, x, config):
    # Extract features from teacher model
    with torch.no_grad():
        if hasattr(teacher_model, "forward") and len(teacher_model.forward(x).shape) == 2:
            teacher_outputs = teacher_model(x)
            if len(teacher_outputs) > 4:  # Has intermediate features
                return teacher_outputs[4]  # final_fea
        else:
            teacher_outputs = teacher_model(x)
            if len(teacher_outputs) > 4:
                return teacher_outputs[4]  # final_fea
    return None

def extract_student_features(student_model, x, config):
    # Extract features from student model
    if config["model"] == "MultiBranchNet":
        logits, features = student_model(x)
        return features[-1]  # Final feature map
    else:
        outputs = student_model(x)
        if len(outputs) > 4:
            return outputs[4]  # final_fea
    return None
```

### Feature Alignment

Features are aligned using adaptive pooling:
```python
if student_features.shape != teacher_features.shape:
    # Resize student features to match teacher features
    student_features = F.adaptive_avg_pool2d(
        student_features, teacher_features.shape[2:]
    )

# Compute MSE loss between features
feature_loss = F.mse_loss(student_features, teacher_features)
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
- Detailed loss components (classification, KD, feature)
- Feature matching metrics

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
- FitNets works best with teacher models that have intermediate feature outputs
- Feature matching helps align student representations with teacher representations
- Knowledge distillation provides additional supervision when teacher is available
