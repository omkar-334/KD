#!/usr/bin/env python3
"""
Example usage for FitNets training
This script demonstrates how to use the FitNets main.py training functions
"""

import os
import sys

# Add parent directory to path
sys.path.append("/Users/omkarkabde/Desktop/KD")


def example_basic_training():
    """Example: Basic FitNets training without teacher"""
    print("=== Example 1: Basic FitNets Training (No Teacher) ===")

    from FitNets.main import create_fitnets_config, train_fitnets

    # Create configuration for basic training
    config = create_fitnets_config(
        model="MultiBranchNet",
        data_dir="../data",
        epochs=5,
        batch_size=16,
        beta=1e-3,  # Feature matching weight
        experiment_name="FitNets_MultiBranchNet_basic",
    )

    print(f"Configuration: {config}")
    print("Starting training...")

    # Train the model
    results = train_fitnets(config)

    print("Training completed!")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Experiment name: {results['experiment_name']}")

    return results


def example_with_teacher():
    """Example: FitNets training with teacher model"""
    print("\n=== Example 2: FitNets Training with Teacher ===")

    from FitNets.main import create_fitnets_config, train_fitnets

    # Create configuration with teacher
    config = create_fitnets_config(
        model="ResNet18",
        teacher_model="../checkpoints/teacher_model.pth",  # Path to teacher
        data_dir="../data",
        epochs=5,
        batch_size=16,
        temperature=3.0,
        alpha=0.1,  # Enable knowledge distillation
        beta=1e-3,  # Feature matching weight
        experiment_name="FitNets_ResNet18_with_teacher",
    )

    print(f"Configuration: {config}")
    print("Starting training with teacher...")

    # Train the model
    results = train_fitnets(config)

    print("Training completed!")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Teacher model used: {results['teacher_model'] is not None}")

    return results


def example_custom_data():
    """Example: FitNets training with custom data loaders"""
    print("\n=== Example 3: FitNets Training with Custom Data ===")

    from FitNets.main import create_fitnets_config, train_fitnets
    from model import prepare_dataloaders

    # Prepare custom data loaders
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        "../data", batch_size=16, augment=True
    )

    # Create configuration
    config = create_fitnets_config(
        model="MultiBranchNet",
        epochs=5,
        beta=1e-3,
        experiment_name="FitNets_custom_data",
    )

    print("Starting training with custom data...")

    # Train with custom data loaders
    results = train_fitnets(config, train_loader, val_loader, test_loader)

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
    checkpoint_path = "../checkpoints/FitNets_MultiBranchNet_best.pth"

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


def example_feature_visualization():
    """Example: Visualize feature maps"""
    print("\n=== Example 5: Feature Map Visualization ===")

    import torch

    from FitNets.main import visualize_feature_maps
    from shared_utils import load_model

    # Load a trained model
    checkpoint_path = "../checkpoints/FitNets_MultiBranchNet_best.pth"

    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = load_model(checkpoint_path)
        model = checkpoint["model"]
        config = checkpoint["config"]

        # Create sample input
        x = torch.randn(1, 3, 128, 128)
        print(f"Input shape: {x.shape}")

        # Visualize feature maps from different layers
        print("Extracting feature maps...")
        for layer_idx in range(4):
            try:
                feature_map = visualize_feature_maps(model, x, config, layer_idx)
                print(f"Layer {layer_idx} feature map shape: {feature_map.shape}")
            except Exception as e:
                print(f"Could not extract feature map from layer {layer_idx}: {e}")

    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train a model first using example_basic_training()")


def example_different_models():
    """Example: FitNets with different model architectures"""
    print("\n=== Example 6: FitNets with Different Models ===")

    from FitNets.main import create_fitnets_config, train_fitnets

    models = ["MultiBranchNet", "ResNet18", "ResNet50"]

    for model_name in models:
        print(f"\n--- Training {model_name} ---")
        config = create_fitnets_config(
            model=model_name,
            data_dir="../data",
            epochs=3,  # Shorter for demo
            batch_size=16,
            beta=1e-3,
            experiment_name=f"FitNets_{model_name}_demo",
        )

        try:
            results = train_fitnets(config)
            print(f"{model_name} - Best accuracy: {results['best_accuracy']:.4f}")
        except Exception as e:
            print(f"Error training {model_name}: {e}")


def example_hyperparameter_tuning():
    """Example: FitNets with different hyperparameters"""
    print("\n=== Example 7: FitNets Hyperparameter Tuning ===")

    from FitNets.main import create_fitnets_config, train_fitnets

    # Test different alpha and beta values
    alpha_values = [0.05, 0.1, 0.2]
    beta_values = [1e-4, 1e-3, 1e-2]

    for alpha in alpha_values:
        for beta in beta_values:
            print(f"\n--- Alpha: {alpha}, Beta: {beta} ---")
            config = create_fitnets_config(
                model="MultiBranchNet",
                data_dir="../data",
                epochs=3,  # Shorter for demo
                batch_size=16,
                alpha=alpha,
                beta=beta,
                experiment_name=f"FitNets_alpha{alpha}_beta{beta}",
            )

            try:
                results = train_fitnets(config)
                print(
                    f"Alpha: {alpha}, Beta: {beta} - Accuracy: {results['best_accuracy']:.4f}"
                )
            except Exception as e:
                print(f"Error with alpha={alpha}, beta={beta}: {e}")


def main():
    """Run all examples"""
    print("FitNets Training Examples")
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

        # Example 5: Feature visualization
        example_feature_visualization()

        # Example 6: Different models
        example_different_models()

        # Example 7: Hyperparameter tuning
        example_hyperparameter_tuning()

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
