#!/usr/bin/env python3
"""
Example usage for BYOT (Bring Your Own Teacher) training
This script demonstrates how to use the BYOT main.py training functions
"""

import os
import sys

# Add parent directory to path
sys.path.append("/Users/omkarkabde/Desktop/KD")


def example_basic_training():
    """Example: Basic BYOT training"""
    print("=== Example 1: Basic BYOT Training ===")

    from BYOT.main_new import create_byot_config, train_byot

    # Create configuration for basic training
    config = create_byot_config(
        model="MultiBranchNet",
        data_dir="../data",
        epochs=5,
        batch_size=16,
        alpha=0.1,  # Knowledge distillation weight
        beta=1e-6,  # Feature matching weight
        experiment_name="BYOT_MultiBranchNet_basic",
    )

    print(f"Configuration: {config}")
    print("Starting training...")

    # Train the model
    results = train_byot(config)

    print("Training completed!")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Experiment name: {results['experiment_name']}")

    return results


def example_resnet_training():
    """Example: BYOT training with ResNet"""
    print("\n=== Example 2: BYOT Training with ResNet ===")

    from BYOT.main_new import create_byot_config, train_byot

    # Create configuration with ResNet
    config = create_byot_config(
        model="ResNet50",
        data_dir="../data",
        epochs=5,
        batch_size=16,
        alpha=0.1,  # Knowledge distillation weight
        beta=1e-6,  # Feature matching weight
        experiment_name="BYOT_ResNet50",
    )

    print(f"Configuration: {config}")
    print("Starting ResNet training...")

    # Train the model
    results = train_byot(config)

    print("Training completed!")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Model: {config['model']}")

    return results


def example_custom_data():
    """Example: BYOT training with custom data loaders"""
    print("\n=== Example 3: BYOT Training with Custom Data ===")

    from BYOT.main_new import create_byot_config, train_byot
    from model import prepare_dataloaders

    # Prepare custom data loaders
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        "../data", batch_size=16, augment=True
    )

    # Create configuration
    config = create_byot_config(
        model="MultiBranchNet",
        epochs=5,
        alpha=0.1,
        beta=1e-6,
        experiment_name="BYOT_custom_data",
    )

    print("Starting training with custom data...")

    # Train with custom data loaders
    results = train_byot(config, train_loader, val_loader, test_loader)

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
    checkpoint_path = "../checkpoints/BYOT_MultiBranchNet_best.pth"

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


def example_different_models():
    """Example: BYOT with different model architectures"""
    print("\n=== Example 5: BYOT with Different Models ===")

    from BYOT.main_new import create_byot_config, train_byot

    models = ["MultiBranchNet", "ResNet18", "ResNet50"]

    for model_name in models:
        print(f"\n--- Training {model_name} ---")
        config = create_byot_config(
            model=model_name,
            data_dir="../data",
            epochs=3,  # Shorter for demo
            batch_size=16,
            alpha=0.1,
            beta=1e-6,
            experiment_name=f"BYOT_{model_name}_demo",
        )

        try:
            results = train_byot(config)
            print(f"{model_name} - Best accuracy: {results['best_accuracy']:.4f}")
        except Exception as e:
            print(f"Error training {model_name}: {e}")


def example_hyperparameter_tuning():
    """Example: BYOT with different hyperparameters"""
    print("\n=== Example 6: BYOT Hyperparameter Tuning ===")

    from BYOT.main_new import create_byot_config, train_byot

    # Test different alpha and beta values
    alpha_values = [0.05, 0.1, 0.2]
    beta_values = [1e-7, 1e-6, 1e-5]

    for alpha in alpha_values:
        for beta in beta_values:
            print(f"\n--- Alpha: {alpha}, Beta: {beta} ---")
            config = create_byot_config(
                model="MultiBranchNet",
                data_dir="../data",
                epochs=3,  # Shorter for demo
                batch_size=16,
                alpha=alpha,
                beta=beta,
                experiment_name=f"BYOT_alpha{alpha}_beta{beta}",
            )

            try:
                results = train_byot(config)
                print(
                    f"Alpha: {alpha}, Beta: {beta} - Accuracy: {results['best_accuracy']:.4f}"
                )
            except Exception as e:
                print(f"Error with alpha={alpha}, beta={beta}: {e}")


def main():
    """Run all examples"""
    print("BYOT (Bring Your Own Teacher) Training Examples")
    print("=" * 50)

    try:
        # Example 1: Basic training
        results1 = example_basic_training()

        # Example 2: ResNet training
        results2 = example_resnet_training()

        # Example 3: Custom data
        results3 = example_custom_data()

        # Example 4: Model evaluation
        example_model_evaluation()

        # Example 5: Different models
        example_different_models()

        # Example 6: Hyperparameter tuning
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
