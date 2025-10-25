# Bring Your Own Teacher (BYOT) Knowledge Distillation

This implementation provides modular functions for BYOT (Bring Your Own Teacher) knowledge distillation that can be easily imported and used in Jupyter notebooks or other Python scripts. BYOT uses intermediate supervision, knowledge distillation, and feature matching to improve student model performance.

## Features

- **Modular Design**: Import and use individual functions in Jupyter notebooks
- **Intermediate Supervision**: Supervise intermediate layers during training
- **Knowledge Distillation**: Distill knowledge from final to intermediate outputs
- **Feature Matching**: Match features between intermediate and final layers
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
from BYOT.main_new import train_byot, create_byot_config

# Create custom configuration
config = create_byot_config(
    model='MultiBranchNet',
    data_dir='./data',
    epochs=50,
    batch_size=32,
    alpha=0.1,  # KD weight
    beta=1e-6   # Feature matching weight
)

# Train the model
results = train_byot(config)

# Access results
model = results['model']
best_accuracy = results['best_accuracy']
training_history = results['training_history']
```

### Using with Different Models

```python
from BYOT.main_new import train_byot, create_byot_config

# Train ResNet50 with BYOT
config = create_byot_config(
    model='ResNet50',
    data_dir='./data',
    epochs=50,
    alpha=0.1,
    beta=1e-6
)

results = train_byot(config)
```

## Available Functions

### Core Training Functions

#### `train_byot(config=None, train_loader=None, val_loader=None, test_loader=None)`
Main training function that handles the complete BYOT training process.

**Parameters:**
- `config` (dict): Training configuration. If None, uses DEFAULT_CONFIG
- `train_loader`: Training data loader (optional if config['data_dir'] is provided)
- `val_loader`: Validation data loader (optional if config['data_dir'] is provided)
- `test_loader`: Test data loader (optional)

**Returns:**
- Dictionary containing:
  - `model`: Trained model
  - `best_accuracy`: Best validation accuracy
  - `training_history`: List of epoch results
  - `test_results`: Test set results (if test_loader provided)
  - `config`: Final configuration used
  - `experiment_name`: Name of the experiment

#### `create_byot_config(**kwargs)`
Create a BYOT configuration dictionary with custom parameters.

**Parameters:**
- `**kwargs`: Configuration parameters to override defaults

**Returns:**
- Dictionary with configuration parameters

### Utility Functions

#### `compute_byot_loss(model, x, y, config)`
Compute BYOT loss for different model architectures.

#### `train_epoch(model, train_loader, optimizer, device, config)`
Train for one epoch.

#### `validate(model, val_loader, device, config)`
Validate the model.

#### `kd_loss_function(student_output, teacher_output, temperature)`
Compute knowledge distillation loss.

#### `feature_loss_function(student_feat, teacher_feat)`
Compute feature matching loss.

## Configuration Parameters

### Default Configuration
```python
DEFAULT_CONFIG = {
    'model': 'MultiBranchNet',              # Model architecture
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
    'beta': 1e-6,                           # Weight for feature matching
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

### 1. Basic BYOT Training
```python
from BYOT.main_new import train_byot, create_byot_config

# Train MultiBranchNet with BYOT
config = create_byot_config(
    model='MultiBranchNet',
    data_dir='./data',
    epochs=50,
    alpha=0.1,
    beta=1e-6
)

results = train_byot(config)
print(f"Best accuracy: {results['best_accuracy']:.4f}")
```

### 2. Training with ResNet
```python
from BYOT.main_new import train_byot, create_byot_config

# Train ResNet50 with BYOT
config = create_byot_config(
    model='ResNet50',
    data_dir='./data',
    epochs=50,
    alpha=0.1,
    beta=1e-6
)

results = train_byot(config)
```

### 3. Custom Data Loaders
```python
from model import prepare_dataloaders
from BYOT.main_new import train_byot, create_byot_config

# Prepare custom data loaders
train_loader, val_loader, test_loader = prepare_dataloaders(
    'data', batch_size=32, augment=True
)

# Train with custom data
config = create_byot_config(model='MultiBranchNet', epochs=50)
results = train_byot(config, train_loader, val_loader, test_loader)
```

### 4. Load and Evaluate Model
```python
from shared_utils import load_model, evaluate_model
from BYOT.main_new import create_byot_config

# Load trained model
checkpoint = load_model('checkpoints/model_best.pth')
model = checkpoint['model']
config = checkpoint['config']

# Evaluate
test_results = evaluate_model(model, test_loader, config=config)
print(f"Test accuracy: {test_results['accuracy']:.4f}")
```

### 5. Hyperparameter Tuning
```python
from BYOT.main_new import train_byot, create_byot_config

# Test different hyperparameters
alpha_values = [0.05, 0.1, 0.2]
beta_values = [1e-7, 1e-6, 1e-5]

for alpha in alpha_values:
    for beta in beta_values:
        config = create_byot_config(
            model='MultiBranchNet',
            data_dir='./data',
            epochs=20,
            alpha=alpha,
            beta=beta,
            experiment_name=f'BYOT_alpha{alpha}_beta{beta}'
        )
        
        results = train_byot(config)
        print(f"Alpha: {alpha}, Beta: {beta} - Accuracy: {results['best_accuracy']:.4f}")
```

## Model Architectures

### Supported Models
- **MultiBranchNet**: Custom multi-branch network with ECA layers
- **ResNet18**: ResNet-18 with intermediate outputs
- **ResNet50**: ResNet-50 with intermediate outputs

All models are designed to work with your `prepare_dataloaders` function and support intermediate feature extraction for BYOT training.

## BYOT Details

The BYOT method implements:

1. **Intermediate Supervision**: Supervise intermediate layers during training
2. **Knowledge Distillation**: Distill knowledge from final to intermediate outputs
3. **Feature Matching**: Match features between intermediate and final layers
4. **Multi-Scale Learning**: Learn at multiple scales simultaneously

### Loss Function

The BYOT method uses:
```
Total Loss = (1-α) × (Final + Intermediate) + α × KD + β × Feature Matching
```

Where:
- **Final Loss**: Cross-entropy loss on final output
- **Intermediate Loss**: Cross-entropy loss on intermediate outputs
- **KD Loss**: Knowledge distillation loss from final to intermediate
- **Feature Loss**: Feature matching loss between intermediate and final features
- **α**: Weight for knowledge distillation
- **β**: Weight for feature matching

### MultiBranchNet Architecture

For MultiBranchNet, the method:
1. Uses all branch outputs for intermediate supervision
2. Distills knowledge from final branch to intermediate branches
3. Matches features between intermediate and final branches
4. Applies ECA attention to each branch

### ResNet Architecture

For ResNet models, the method:
1. Uses intermediate outputs (middle_output1, middle_output2, middle_output3)
2. Distills knowledge from final output to intermediate outputs
3. Matches features between intermediate and final features
4. Applies feature matching loss

## Monitoring Training

Use TensorBoard to monitor training:

```python
# TensorBoard logs are automatically saved
# View with: tensorboard --logdir logs/
```

Available metrics:
- Training/Validation loss and accuracy
- Learning rate schedule
- Detailed loss components (final, intermediate, KD, feature)
- Loss breakdown by component

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
- BYOT works best with models that have multiple intermediate outputs
- Feature matching helps align intermediate representations with final features
- Knowledge distillation helps transfer knowledge from final to intermediate layers
