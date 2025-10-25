#!/usr/bin/env python3
"""
Example usage for DML (Deep Mutual Learning) training
This script demonstrates how to use the DML main.py training functions
"""

import os
import sys

# Add parent directory to path
sys.path.append("/Users/omkarkabde/Desktop/KD")


def example_basic_dml_training():
    """Example: Basic DML training with 2 models"""
    print("=== Example 1: Basic DML Training (2 Models) ===")

    from DML.main import create_dml_config, train_dml

    # Create configuration for basic DML training
    config = create_dml_config(
        model="ResNet32",
        data_dir="../data",
        epochs=50,
        batch_size=32,
        model_num=2,  # Number of models for DML
        lr=0.1,
        experiment_name="DML_ResNet32_basic",
    )

    print(f"Configuration: {config}")
    print("Starting DML training...")

    # Train the models
    results = train_dml(config)

    print("Training completed!")
    print(f"Individual accuracies: {results['best_accuracies']}")
    print(f"Average best accuracy: {results['avg_best_accuracy']:.4f}")
    print(f"Experiment name: {results['experiment_name']}")

    return results


def example_multi_model_dml():
    """Example: DML training with 3 models"""
    print("\n=== Example 2: Multi-Model DML Training (3 Models) ===")

    from DML.main import create_dml_config, train_dml

    # Create configuration with 3 models
    config = create_dml_config(
        model="ResNet56",
        data_dir="../data",
        epochs=50,
        batch_size=32,
        model_num=3,  # Train 3 models
        lr=0.1,
        experiment_name="DML_ResNet56_multi",
    )

    print(f"Configuration: {config}")
    print("Starting multi-model DML training...")

    # Train the models
    results = train_dml(config)

    print("Training completed!")
    print(f"Individual accuracies: {results['best_accuracies']}")
    print(f"Average best accuracy: {results['avg_best_accuracy']:.4f}")
    print(f"Number of models trained: {len(results['models'])}")

    return results


def example_custom_data_dml():
    """Example: DML training with custom data loaders"""
    print("\n=== Example 3: DML Training with Custom Data ===")

    from DML.main import create_dml_config, train_dml
    from model import prepare_dataloaders

    # Prepare custom data loaders
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        "../data", batch_size=32, augment=True
    )

    # Create configuration
    config = create_dml_config(
        model="ResNet44",
        epochs=50,
        model_num=2,
        lr=0.1,
        experiment_name="DML_custom_data",
    )

    print("Starting DML training with custom data...")

    # Train with custom data loaders
    results = train_dml(config, train_loader, val_loader, test_loader)

    print("Training completed!")
    print(f"Individual accuracies: {results['best_accuracies']}")
    print(f"Average best accuracy: {results['avg_best_accuracy']:.4f}")
    print(f"Test results: {results['test_results']}")

    return results


def example_model_evaluation():
    """Example: Load and evaluate DML models"""
    print("\n=== Example 4: DML Model Evaluation ===")

    from DML.main import evaluate_dml_models, load_dml_models
    from model import prepare_dataloaders

    # Load trained models
    checkpoint_paths = [
        "../checkpoints/DML_ResNet32_basic_model1_best.pth",
        "../checkpoints/DML_ResNet32_basic_model2_best.pth",
    ]

    if all(os.path.exists(path) for path in checkpoint_paths):
        print("Loading models from checkpoints...")
        models_data = load_dml_models(checkpoint_paths)
        models = models_data["models"]

        print(f"Loaded {len(models)} models")

        # Prepare test data
        _, _, test_loader = prepare_dataloaders("../data", batch_size=32)

        # Evaluate individual models and ensemble
        print("Evaluating models...")
        test_results = evaluate_dml_models(models, test_loader)

        print("Evaluation results:")
        for result in test_results:
            if result["model_idx"] == "ensemble":
                print(f"Ensemble: {result['accuracy']:.4f}")
            else:
                print(f"Model {result['model_idx'] + 1}: {result['accuracy']:.4f}")

    else:
        print(
            "Checkpoints not found. Please train models first using example_basic_dml_training()"
        )


def example_ensemble_evaluation():
    """Example: Evaluate ensemble performance"""
    print("\n=== Example 5: Ensemble Evaluation ===")

    from DML.main import evaluate_ensemble, load_dml_models
    from model import prepare_dataloaders

    # Load trained models
    checkpoint_paths = [
        "../checkpoints/DML_ResNet32_basic_model1_best.pth",
        "../checkpoints/DML_ResNet32_basic_model2_best.pth",
    ]

    if all(os.path.exists(path) for path in checkpoint_paths):
        print("Loading models for ensemble evaluation...")
        models_data = load_dml_models(checkpoint_paths)
        models = models_data["models"]

        # Prepare test data
        _, _, test_loader = prepare_dataloaders("../data", batch_size=32)

        # Evaluate ensemble
        print("Evaluating ensemble...")
        ensemble_result = evaluate_ensemble(models, test_loader)

        print(f"Ensemble accuracy: {ensemble_result['accuracy']:.4f}")
        print(f"Ensemble loss: {ensemble_result['loss']:.4f}")

    else:
        print(
            "Checkpoints not found. Please train models first using example_basic_dml_training()"
        )


def example_different_architectures():
    """Example: DML with different ResNet architectures"""
    print("\n=== Example 6: DML with Different Architectures ===")

    from DML.main import create_dml_config, train_dml

    architectures = ["ResNet20", "ResNet32", "ResNet44", "ResNet56"]

    for arch in architectures:
        print(f"\n--- Training {arch} ---")
        config = create_dml_config(
            model=arch,
            data_dir="../data",
            epochs=20,  # Shorter for demo
            batch_size=32,
            model_num=2,
            lr=0.1,
            experiment_name=f"DML_{arch}_demo",
        )

        try:
            results = train_dml(config)
            print(f"{arch} - Average accuracy: {results['avg_best_accuracy']:.4f}")
        except Exception as e:
            print(f"Error training {arch}: {e}")


def example_training_comparison():
    """Example: Compare DML with different numbers of models"""
    print("\n=== Example 7: DML Model Count Comparison ===")

    from DML.main import create_dml_config, train_dml

    model_counts = [2, 3, 4]
    results_comparison = []

    for num_models in model_counts:
        print(f"\n--- Training with {num_models} models ---")
        config = create_dml_config(
            model="ResNet32",
            data_dir="../data",
            epochs=30,  # Shorter for demo
            batch_size=32,
            model_num=num_models,
            lr=0.1,
            experiment_name=f"DML_ResNet32_{num_models}models",
        )

        try:
            results = train_dml(config)
            results_comparison.append({
                "num_models": num_models,
                "avg_accuracy": results["avg_best_accuracy"],
                "individual_accuracies": results["best_accuracies"],
            })
            print(
                f"{num_models} models - Average accuracy: {results['avg_best_accuracy']:.4f}"
            )
        except Exception as e:
            print(f"Error training with {num_models} models: {e}")

    # Print comparison
    print("\n--- Comparison Results ---")
    for result in results_comparison:
        print(f"{result['num_models']} models: {result['avg_accuracy']:.4f}")


def main():
    """Run all examples"""
    print("DML (Deep Mutual Learning) Training Examples")
    print("=" * 50)

    try:
        # Example 1: Basic DML training
        results1 = example_basic_dml_training()

        # Example 2: Multi-model DML
        results2 = example_multi_model_dml()

        # Example 3: Custom data
        results3 = example_custom_data_dml()

        # Example 4: Model evaluation
        example_model_evaluation()

        # Example 5: Ensemble evaluation
        example_ensemble_evaluation()

        # Example 6: Different architectures
        example_different_architectures()

        # Example 7: Model count comparison
        example_training_comparison()

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
