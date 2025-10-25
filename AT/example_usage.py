#!/usr/bin/env python3
"""
Example usage for AT (Attention Transfer) training
This script demonstrates how to use the AT main.py training functions
"""

import os
import sys

# Add parent directory to path
sys.path.append("/Users/omkarkabde/Desktop/KD")


def example_basic_training():
    """Example: Basic AT training without teacher"""
    print("=== Example 1: Basic AT Training (No Teacher) ===")

    from AT.main import create_at_config, train_at

    # Create configuration for basic training
    config = create_at_config(
        model="MultiBranchNet",
        data_dir="../data",
        epochs=5,
        batch_size=16,
        beta=1e3,  # Attention transfer weight
        experiment_name="AT_MultiBranchNet_basic",
    )

    print(f"Configuration: {config}")
    print("Starting training...")

    # Train the model
    results = train_at(config)

    print("Training completed!")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Experiment name: {results['experiment_name']}")

    return results


def example_with_teacher():
    """Example: AT training with teacher model"""
    print("\n=== Example 2: AT Training with Teacher ===")

    from AT.main import create_at_config, train_at
    from teacher import get_teacher_path

    # First, train a teacher model if it doesn't exist
    teacher_path = get_teacher_path("ResNet50")
    print(f"Teacher path: {teacher_path}")

    # Create configuration with teacher
    config = create_at_config(
        model="ResNet18",
        data_dir="../data",
        epochs=5,
        batch_size=16,
        temperature=4.0,
        alpha=0.1,  # Enable knowledge distillation
        beta=1e3,  # Attention transfer weight
        experiment_name="AT_ResNet18_with_teacher",
    )

    print(f"Configuration: {config}")
    print("Starting training with teacher...")

    # Train the model
    results = train_at(config)

    print("Training completed!")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Teacher model used: {results['teacher_model'] is not None}")

    return results


def example_custom_data():
    """Example: AT training with custom data loaders"""
    print("\n=== Example 3: AT Training with Custom Data ===")

    from AT.main import create_at_config, train_at
    from model import prepare_dataloaders

    # Prepare custom data loaders
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        "../data", batch_size=16, augment=True
    )

    # Create configuration
    config = create_at_config(
        model="MultiBranchNet", epochs=5, beta=1e3, experiment_name="AT_custom_data"
    )

    print("Starting training with custom data...")

    # Train with custom data loaders
    results = train_at(config, train_loader, val_loader, test_loader)

    print("Training completed!")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Test results: {results['test_results']}")

    return results


def example_model_evaluation():
    """Example: Load and evaluate a trained model"""
    print("\n=== Example 4: Model Evaluation ===")

    from model import prepare_dataloaders
    from shared_utils import evaluate_model, load_model

    # Load a trained model
    checkpoint_path = "../checkpoints/AT_MultiBranchNet_best.pth"

    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = load_model(checkpoint_path)
        model = checkpoint["model"]
        config = checkpoint["config"]

        print(f"Model loaded from epoch {checkpoint['epoch']}")
        print(f"Best accuracy: {checkpoint['best_acc']:.4f}")

        # Prepare test data
        _, _, test_loader = prepare_dataloaders("../data", batch_size=16)

        # Evaluate the model
        print("Evaluating model...")
        test_results = evaluate_model(model, test_loader, config=config)

        print(f"Test accuracy: {test_results['accuracy']:.4f}")
        print(f"Test loss: {test_results['loss']:.4f}")

    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train a model first using example_basic_training()")


def example_attention_visualization():
    """Example: Visualize attention maps"""
    print("\n=== Example 5: Attention Map Visualization ===")

    import torch

    from AT.main_new import visualize_attention_maps
    from shared_utils import load_model

    # Load a trained model
    checkpoint_path = "../checkpoints/AT_MultiBranchNet_best.pth"

    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = load_model(checkpoint_path)
        model = checkpoint["model"]
        config = checkpoint["config"]

        # Create sample input
        x = torch.randn(1, 3, 128, 128)
        print(f"Input shape: {x.shape}")

        # Visualize attention maps from different layers
        print("Extracting attention maps...")
        for layer_idx in range(4):
            try:
                attention_map = visualize_attention_maps(model, x, config, layer_idx)
                print(f"Layer {layer_idx} attention map shape: {attention_map.shape}")
            except Exception as e:
                print(f"Could not extract attention map from layer {layer_idx}: {e}")

    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train a model first using example_basic_training()")


def main():
    """Run all examples"""
    print("AT (Attention Transfer) Training Examples")
    print("=" * 50)

    try:
        # Example 1: Basic training
        results1 = example_basic_training()

        # Example 2: Training with teacher (if teacher model exists)
        if os.path.exists("../checkpoints/teacher_model.pth"):
            results2 = example_with_teacher()
        else:
            print("\n=== Example 2: Skipped (No teacher model found) ===")
            print("Teacher model not found at '../checkpoints/teacher_model.pth'")
            print("Skipping teacher training example...")

        # Example 3: Custom data
        results3 = example_custom_data()

        # Example 4: Model evaluation
        example_model_evaluation()

        # Example 5: Attention visualization
        example_attention_visualization()

        print("\n" + "=" * 50)
        print("All examples completed!")
        print("Check the 'checkpoints' and 'logs' directories for results.")
        print("Use 'tensorboard --logdir logs' to view training metrics.")

    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install torch torchvision tensorboard tqdm")


if __name__ == "__main__":
    main()
