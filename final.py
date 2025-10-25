#!/usr/bin/env python3
"""
Final comprehensive script for Knowledge Distillation comparison
1. Prepare dataloaders
2. Train ResNet50 teacher
3. Run all knowledge distillation methods (AT, BYOT, DML, FitNets)
4. Store metrics and compare results
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

import pandas as pd

# Add project root to path
sys.path.append("/Users/omkarkabde/Desktop/KD")

# Import all KD methods
from AT.main import create_at_config, train_at
from BYOT.main import create_byot_config, train_byot
from DML.main import create_dml_config, train_dml
from FitNets.main import create_fitnets_config, train_fitnets
from shared_utils import prepare_data_loaders
from teacher import create_teacher_config, train_teacher


def setup_experiment_logging():
    """Setup logging for the final experiment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/final_experiment_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "final_experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return log_dir


def prepare_dataloaders(data_dir="data", batch_size=32, dataset_name="default"):
    """Prepare data loaders for all experiments"""
    logging.info("=== Preparing Data Loaders ===")

    # Common data configuration
    data_config = {
        "data_dir": data_dir,
        "batch_size": batch_size,
        "num_workers": 4,
        "augment": True,
    }

    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(data_config)

    num_classes = len(train_loader.dataset.dataset.classes)
    logging.info("Dataset prepared: %s classes", num_classes)
    logging.info(f"Train samples: {len(train_loader.dataset)}")
    logging.info(f"Val samples: {len(val_loader.dataset)}")
    logging.info(f"Test samples: {len(test_loader.dataset)}")

    return train_loader, val_loader, test_loader, num_classes, data_config, dataset_name


def train_teacher_model(data_config, num_classes, log_dir, dataset_name):
    """Train ResNet50 teacher model"""
    logging.info("=== Training ResNet50 Teacher Model ===")

    # Create teacher configuration
    teacher_config = create_teacher_config(
        model="ResNet50",
        data_dir=data_config["data_dir"],
        batch_size=data_config["batch_size"],
        epochs=100,  # Train longer for better teacher
        lr=0.1,
        step_size=30,
        gamma=0.1,
        device="auto",
        experiment_name="ResNet50_Teacher_Final",
        save_dir="checkpoints",
        log_dir="logs",
    )

    # Train teacher
    start_time = time.time()
    teacher_results = train_teacher(teacher_config, dataset_name=dataset_name)
    training_time = time.time() - start_time

    logging.info(f"Teacher training completed in {training_time:.2f} seconds")
    logging.info(f"Teacher best accuracy: {teacher_results['best_accuracy']:.4f}")

    # Save teacher results
    teacher_metrics = {
        "method": "Teacher",
        "model": "ResNet50",
        "best_accuracy": teacher_results["best_accuracy"],
        "training_time": training_time,
        "epochs": teacher_config["epochs"],
        "final_test_acc": teacher_results.get("test_results", {}).get("test_acc", None),
        "final_test_loss": teacher_results.get("test_results", {}).get(
            "test_loss", None
        ),
    }

    return teacher_results, teacher_metrics


def run_knowledge_distillation_methods(data_config, num_classes, log_dir, dataset_name):
    """Run all knowledge distillation methods"""
    logging.info("=== Running Knowledge Distillation Methods ===")

    all_results = []

    # Common configuration for all methods
    common_config = {
        "data_dir": data_config["data_dir"],
        "batch_size": data_config["batch_size"],
        "num_workers": data_config["num_workers"],
        "augment": data_config["augment"],
        "device": "auto",
    }

    # 1. AT (Attention Transfer)
    logging.info("--- Training AT (Attention Transfer) ---")
    try:
        at_config = create_at_config(
            model="ResNet18",
            data_dir=common_config["data_dir"],
            batch_size=common_config["batch_size"],
            epochs=30,
            lr=0.01,
            temperature=4.0,
            alpha=0.1,
            beta=1e3,
            experiment_name="AT_ResNet18_Final",
        )

        start_time = time.time()
        at_results = train_at(at_config, dataset_name=dataset_name)
        training_time = time.time() - start_time

        at_metrics = {
            "method": "AT",
            "model": "ResNet18",
            "teacher": "ResNet50",
            "best_accuracy": at_results["best_accuracy"],
            "training_time": training_time,
            "epochs": at_config["epochs"],
            "final_test_acc": at_results.get("test_results", {}).get("test_acc", None),
            "final_test_loss": at_results.get("test_results", {}).get(
                "test_loss", None
            ),
        }
        all_results.append(at_metrics)
        logging.info(f"AT completed: {at_results['best_accuracy']:.4f} accuracy")

    except Exception as e:
        logging.exception("AT training failed: %s", e)
        all_results.append({
            "method": "AT",
            "model": "ResNet18",
            "teacher": "ResNet50",
            "best_accuracy": 0.0,
            "training_time": 0.0,
            "epochs": 50,
            "error": str(e),
        })

    # 2. BYOT (Bring Your Own Teacher)
    logging.info("--- Training BYOT (Bring Your Own Teacher) ---")
    try:
        byot_config = create_byot_config(
            model="ResNet18",
            data_dir=common_config["data_dir"],
            batch_size=common_config["batch_size"],
            epochs=30,
            lr=0.01,
            temperature=4.0,
            alpha=0.1,
            beta=0.1,
            experiment_name="BYOT_ResNet18_Final",
        )

        start_time = time.time()
        byot_results = train_byot(byot_config, dataset_name=dataset_name)
        training_time = time.time() - start_time

        byot_metrics = {
            "method": "BYOT",
            "model": "ResNet18",
            "teacher": "ResNet50",
            "best_accuracy": byot_results["best_accuracy"],
            "training_time": training_time,
            "epochs": byot_config["epochs"],
            "final_test_acc": byot_results.get("test_results", {}).get(
                "test_acc", None
            ),
            "final_test_loss": byot_results.get("test_results", {}).get(
                "test_loss", None
            ),
        }
        all_results.append(byot_metrics)
        logging.info(f"BYOT completed: {byot_results['best_accuracy']:.4f} accuracy")

    except Exception as e:
        logging.exception("BYOT training failed: %s", e)
        all_results.append({
            "method": "BYOT",
            "model": "ResNet18",
            "teacher": "ResNet50",
            "best_accuracy": 0.0,
            "training_time": 0.0,
            "epochs": 50,
            "error": str(e),
        })

    # 3. DML (Deep Mutual Learning)
    logging.info("--- Training DML (Deep Mutual Learning) ---")
    try:
        dml_config = create_dml_config(
            model="ResNet32",  # Use ResNet32 which is supported by DML
            data_dir=common_config["data_dir"],
            batch_size=common_config["batch_size"],
            epochs=30,
            lr=0.01,
            model_num=2,  # Number of student models
            experiment_name="DML_ResNet32_Final",
        )

        start_time = time.time()
        dml_results = train_dml(dml_config, dataset_name=dataset_name)
        training_time = time.time() - start_time

        # DML returns results for multiple models, take the best
        best_acc = (
            max(dml_results["best_accuracies"])
            if "best_accuracies" in dml_results
            else 0.0
        )

        dml_metrics = {
            "method": "DML",
            "model": "ResNet32",
            "teacher": "None (Mutual Learning)",
            "best_accuracy": best_acc,
            "training_time": training_time,
            "epochs": dml_config["epochs"],
            "model_num": dml_config["model_num"],
            "final_test_acc": dml_results.get("test_results", {}).get("test_acc", None),
            "final_test_loss": dml_results.get("test_results", {}).get(
                "test_loss", None
            ),
        }
        all_results.append(dml_metrics)
        logging.info(f"DML completed: {best_acc:.4f} accuracy")

    except Exception as e:
        logging.exception("DML training failed: %s", e)
        all_results.append({
            "method": "DML",
            "model": "ResNet18",
            "teacher": "None (Mutual Learning)",
            "best_accuracy": 0.0,
            "training_time": 0.0,
            "epochs": 50,
            "error": str(e),
        })

    # 4. FitNets
    logging.info("--- Training FitNets ---")
    try:
        fitnets_config = create_fitnets_config(
            model="ResNet18",
            data_dir=common_config["data_dir"],
            batch_size=common_config["batch_size"],
            epochs=30,
            lr=0.01,
            temperature=4.0,
            alpha=0.1,
            beta=1e3,
            experiment_name="FitNets_ResNet18_Final",
        )

        start_time = time.time()
        fitnets_results = train_fitnets(fitnets_config, dataset_name=dataset_name)
        training_time = time.time() - start_time

        fitnets_metrics = {
            "method": "FitNets",
            "model": "ResNet18",
            "teacher": "ResNet50",
            "best_accuracy": fitnets_results["best_accuracy"],
            "training_time": training_time,
            "epochs": fitnets_config["epochs"],
            "final_test_acc": fitnets_results.get("test_results", {}).get(
                "test_acc", None
            ),
            "final_test_loss": fitnets_results.get("test_results", {}).get(
                "test_loss", None
            ),
        }
        all_results.append(fitnets_metrics)
        logging.info(
            f"FitNets completed: {fitnets_results['best_accuracy']:.4f} accuracy"
        )

    except Exception as e:
        logging.exception("FitNets training failed: %s", e)
        all_results.append({
            "method": "FitNets",
            "model": "ResNet18",
            "teacher": "ResNet50",
            "best_accuracy": 0.0,
            "training_time": 0.0,
            "epochs": 50,
            "error": str(e),
        })

    return all_results


def compare_results(teacher_metrics, kd_results, log_dir):
    """Compare and analyze results from all methods"""
    logging.info("=== Comparing Results ===")

    # Combine all results
    if teacher_metrics:
        all_metrics = [teacher_metrics] + kd_results
    else:
        all_metrics = kd_results

    # Create DataFrame for analysis
    df = pd.DataFrame(all_metrics)

    # Sort by accuracy
    df_sorted = df.sort_values("best_accuracy", ascending=False)

    # Print comparison table
    print("\n" + "=" * 80)
    print("KNOWLEDGE DISTILLATION COMPARISON RESULTS")
    print("=" * 80)
    print(df_sorted.to_string(index=False, float_format="%.4f"))
    print("=" * 80)

    # Calculate improvements over teacher (if teacher exists)
    if teacher_metrics:
        teacher_acc = teacher_metrics["best_accuracy"]
        print(f"\nTeacher (ResNet50) Accuracy: {teacher_acc:.4f}")
        print("\nStudent Model Improvements:")
        print("-" * 50)

        for _, row in df_sorted.iterrows():
            if row["method"] != "Teacher":
                improvement = row["best_accuracy"] - teacher_acc
                improvement_pct = (improvement / teacher_acc) * 100
                print(
                    f"{row['method']:12}: {row['best_accuracy']:.4f} "
                    f"({improvement:+.4f}, {improvement_pct:+.2f}%)"
                )
    else:
        print("\nNo teacher model - comparing student methods only")
        print("-" * 50)
        for _, row in df_sorted.iterrows():
            print(f"{row['method']:12}: {row['best_accuracy']:.4f}")

    # Save detailed results
    results_file = os.path.join(log_dir, "comparison_results.json")
    with open(results_file, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    # Save CSV
    csv_file = os.path.join(log_dir, "comparison_results.csv")
    df_sorted.to_csv(csv_file, index=False)

    # Create summary
    summary = {
        "experiment_timestamp": datetime.now().isoformat(),
        "teacher_accuracy": teacher_metrics["best_accuracy"]
        if teacher_metrics
        else None,
        "best_student_method": df_sorted.iloc[0]["method"]
        if len(df_sorted) > 0
        else "None",
        "best_student_accuracy": df_sorted.iloc[0]["best_accuracy"]
        if len(df_sorted) > 0
        else 0.0,
        "total_methods_tested": len(kd_results),
        "successful_methods": len([r for r in kd_results if "error" not in r]),
        "failed_methods": len([r for r in kd_results if "error" in r]),
    }

    summary_file = os.path.join(log_dir, "experiment_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("Results saved to %s", log_dir)
    logging.info(f"Best student method: {summary['best_student_method']}")
    logging.info(f"Best student accuracy: {summary['best_student_accuracy']:.4f}")

    return df_sorted, summary


def main(data_dir="data", dataset_name="default", batch_size=32):
    """Main execution function"""
    print("Starting Knowledge Distillation Comparison Experiment")
    print("=" * 60)
    print(f"Dataset: {dataset_name}")
    print(f"Data directory: {data_dir}")

    # Setup logging
    log_dir = setup_experiment_logging()
    logging.info("Final Knowledge Distillation Comparison Experiment Started")
    logging.info("Dataset: %s", dataset_name)

    try:
        # # 1. Prepare data loaders
        (
            train_loader,
            val_loader,
            test_loader,
            num_classes,
            data_config,
            dataset_name,
        ) = prepare_dataloaders(data_dir, batch_size, dataset_name)

        # # 2. Train ResNet50 teacher
        # teacher_results, teacher_metrics = train_teacher_model(
        #     data_config, num_classes, log_dir, dataset_name
        # )

        # 3. Run all knowledge distillation methods
        kd_results = run_knowledge_distillation_methods(
            data_config, num_classes, log_dir, dataset_name
        )

        # 4. Compare and analyze results
        comparison_df, summary = compare_results(
            teacher_metrics=None, kd_results=kd_results, log_dir=log_dir
        )

        print("\nExperiment completed successfully!")
        print(f"Results saved to: {log_dir}")
        print(f"Best performing method: {summary['best_student_method']}")

        return comparison_df, summary

    except Exception as e:
        logging.exception("Experiment failed: %s", e)
        print(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    # Run the experiment
    comparison_df, summary = main()
