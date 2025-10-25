#!/usr/bin/env python3
"""
Run knowledge distillation experiments on multiple datasets
"""

import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append("/Users/omkarkabde/Desktop/KD")

from final import main as run_experiment


def run_experiment_on_dataset(data_dir, dataset_name, batch_size=32):
    """Run experiment on a single dataset"""
    print(f"\n{'=' * 80}")
    print(f"🚀 Running experiment on dataset: {dataset_name}")
    print(f"📂 Data directory: {data_dir}")
    print(f"{'=' * 80}")

    try:
        comparison_df, summary = run_experiment(data_dir, dataset_name, batch_size)

        print(f"\n✅ Experiment on {dataset_name} completed successfully!")
        print(f"📊 Best method: {summary['best_student_method']}")
        print(f"🎯 Best accuracy: {summary['best_student_accuracy']:.4f}")

        return {
            "dataset": dataset_name,
            "data_dir": data_dir,
            "success": True,
            "best_method": summary["best_student_method"],
            "best_accuracy": summary["best_student_accuracy"],
            "summary": summary,
        }

    except Exception as e:
        print(f"❌ Experiment on {dataset_name} failed: {e}")
        return {
            "dataset": dataset_name,
            "data_dir": data_dir,
            "success": False,
            "error": str(e),
        }


def main():
    """Run experiments on multiple datasets"""
    print("🚀 Multi-Dataset Knowledge Distillation Comparison")
    print("=" * 80)

    # Define your datasets here
    datasets = [
        {"name": "cifar10", "data_dir": "data/cifar10", "batch_size": 32},
        {"name": "cifar100", "data_dir": "data/cifar100", "batch_size": 32},
        {
            "name": "imagenet_subset",
            "data_dir": "data/imagenet_subset",
            "batch_size": 16,  # Smaller batch size for larger images
        },
        # Add more datasets as needed
        # {
        #     "name": "custom_dataset",
        #     "data_dir": "data/custom",
        #     "batch_size": 32
        # }
    ]

    # Filter out datasets that don't exist
    available_datasets = []
    for dataset in datasets:
        if os.path.exists(dataset["data_dir"]):
            available_datasets.append(dataset)
            print(f"✅ Found dataset: {dataset['name']} at {dataset['data_dir']}")
        else:
            print(f"❌ Dataset not found: {dataset['name']} at {dataset['data_dir']}")

    if not available_datasets:
        print("❌ No datasets found! Please check your data directories.")
        return None

    print(f"\n📊 Running experiments on {len(available_datasets)} datasets...")

    # Run experiments
    results = []
    for dataset in available_datasets:
        result = run_experiment_on_dataset(
            dataset["data_dir"], dataset["name"], dataset["batch_size"]
        )
        results.append(result)

    # Print summary
    print(f"\n{'=' * 80}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'=' * 80}")

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"✅ Successful experiments: {len(successful)}")
    print(f"❌ Failed experiments: {len(failed)}")

    if successful:
        print("\n📈 Results by Dataset:")
        print("-" * 50)
        for result in successful:
            print(
                f"{result['dataset']:20}: {result['best_method']:10} - {result['best_accuracy']:.4f}"
            )

    if failed:
        print("\n❌ Failed Datasets:")
        print("-" * 50)
        for result in failed:
            print(f"{result['dataset']:20}: {result['error']}")

    # Save overall results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"logs/multi_dataset_results_{timestamp}.json"

    import json

    os.makedirs("logs", exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📁 Overall results saved to: {results_file}")

    return results


if __name__ == "__main__":
    results = main()
