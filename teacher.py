import logging
import os
import sys
import time

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append("/Users/omkarkabde/Desktop/KD")
from shared_utils import (
    AverageMeter,
    count_parameters,
    create_optimizer,
    create_scheduler,
    evaluate_model,
    get_device,
    get_model,
    log_metrics,
    prepare_data_loaders,
    print_training_info,
    save_checkpoint,
    save_experiment_summary,
    setup_logging,
    setup_tensorboard,
)

# Default configuration for teacher training
TEACHER_CONFIG = {
    "model": "ResNet50",  # Default teacher model
    "data_dir": "data",
    "batch_size": 32,
    "num_workers": 4,
    "augment": True,  # Use augmentation for teacher
    "epochs": 100,  # Train longer for better teacher
    "lr": 0.1,  # Higher learning rate for teacher
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "step_size": 30,  # LR schedule for longer training
    "gamma": 0.1,
    "device": "auto",
    "print_freq": 100,
    "save_freq": 10,
    "save_dir": "checkpoints/teachers",
    "log_dir": "logs/teachers",
    "experiment_name": None,
    "resume": None,
}


def train_epoch(model, train_loader, optimizer, device, config):
    """Train teacher model for one epoch"""
    model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()

    for i, (x, y) in enumerate(tqdm(train_loader, desc="Training Teacher")):
        data_time.update(time.time() - end)

        x, y = x.to(device), y.to(device)

        # Forward pass
        if config["model"] == "MultiBranchNet":
            logits, _ = model(x)
            pred = logits[-1]  # Use final prediction
        else:
            pred = model(x)

        # Compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss.item(), x.size(0))

        # Compute accuracy
        with torch.no_grad():
            correct = (pred.argmax(1) == y).sum().item()
            top1.update(correct / x.size(0) * 100, x.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config["print_freq"] == 0:
            logging.info(
                f"Batch [{i}/{len(train_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})"
            )

    return losses.avg, top1.avg


def validate(model, val_loader, device, config):
    """Validate teacher model"""
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating Teacher"):
            x, y = x.to(device), y.to(device)

            # Forward pass
            if config["model"] == "MultiBranchNet":
                logits, _ = model(x)
                pred = logits[-1]  # Use final prediction
            else:
                pred = model(x)

            # Compute loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(pred, y)

            # Compute accuracy
            correct = (pred.argmax(1) == y).sum().item()

            losses.update(loss.item(), x.size(0))
            top1.update(correct / x.size(0) * 100, x.size(0))

    return losses.avg, top1.avg


def train_teacher(
    config=None,
    train_loader=None,
    val_loader=None,
    test_loader=None,
    dataset_name="default",
):
    """
    Train a teacher model

    Args:
        config (dict): Training configuration. If None, uses TEACHER_CONFIG
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (optional)
        dataset_name (str): Name of the dataset for organizing outputs

    Returns:
        dict: Training results including model, best accuracy, and metrics
    """
    if config is None:
        config = TEACHER_CONFIG.copy()

    # Setup device
    device = get_device(config["device"])

    # Setup logging
    experiment_name = setup_logging(config, "Teacher", dataset_name)
    config["experiment_name"] = experiment_name

    # Setup tensorboard
    writer = setup_tensorboard(config, "Teacher")

    # Prepare data if not provided
    if train_loader is None or val_loader is None:
        logging.info("Preparing data loaders...")
        train_loader, val_loader, test_loader = prepare_data_loaders(config)

    num_classes = len(train_loader.dataset.dataset.classes)
    config["num_classes"] = num_classes

    # Create teacher model
    logging.info(f"Creating {config['model']} teacher model...")
    teacher_model = get_model(config, num_classes)
    teacher_model = teacher_model.to(device)

    # Count parameters
    total_params, trainable_params = count_parameters(teacher_model)
    logging.info(f"Teacher model parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Setup optimizer and scheduler
    optimizer = create_optimizer(teacher_model, config)
    scheduler = create_scheduler(optimizer, config)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0

    if config["resume"]:
        logging.info(f"Resuming from checkpoint: {config['resume']}")
        checkpoint = torch.load(config["resume"], map_location=device)
        teacher_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_acc = checkpoint["best_acc"]
        logging.info(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.4f}")

    # Print training info
    print_training_info(
        config,
        total_params,
        trainable_params,
        num_classes,
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    # Training loop
    logging.info("Starting teacher training...")
    training_history = []

    for epoch in range(start_epoch, config["epochs"]):
        logging.info(f"Epoch {epoch}/{config['epochs'] - 1}")

        # Train
        train_loss, train_acc = train_epoch(
            teacher_model, train_loader, optimizer, device, config
        )

        # Validate
        val_loss, val_acc = validate(teacher_model, val_loader, device, config)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Log metrics
        logging.info(
            f"Epoch {epoch}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"LR: {current_lr:.6f}"
        )

        # Store epoch results
        epoch_results = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        }
        training_history.append(epoch_results)

        # Tensorboard logging
        log_metrics(
            writer,
            {
                "Train/Loss": train_loss,
                "Train/Accuracy": train_acc,
                "Val/Loss": val_loss,
                "Val/Accuracy": val_acc,
                "Learning_Rate": current_lr,
            },
            epoch,
        )

        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        if epoch % config["save_freq"] == 0 or is_best:
            save_checkpoint(teacher_model, optimizer, epoch, best_acc, config, is_best)

    # Final evaluation on test set
    test_results = None
    if test_loader is not None:
        logging.info("Evaluating teacher on test set...")
        test_result = evaluate_model(teacher_model, test_loader, device, config)
        logging.info(
            f"Test Loss: {test_result['loss']:.4f}, Test Acc: {test_result['accuracy']:.4f}"
        )
        log_metrics(
            writer,
            {
                "Test/Loss": test_result["loss"],
                "Test/Accuracy": test_result["accuracy"],
            },
            epoch,
        )
        test_results = {
            "test_loss": test_result["loss"],
            "test_acc": test_result["accuracy"],
        }

    # Save final teacher model
    final_checkpoint = {
        "epoch": config["epochs"],
        "model_state_dict": teacher_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
        "config": config,
    }
    final_path = os.path.join(
        config["save_dir"], f"{config['model']}_teacher_final.pth"
    )
    torch.save(final_checkpoint, final_path)
    logging.info("Final teacher model saved: %s", final_path)

    # Save best teacher model
    best_checkpoint = {
        "epoch": config["epochs"],
        "model_state_dict": teacher_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
        "config": config,
    }
    best_path = os.path.join(config["save_dir"], f"{config['model']}_teacher_best.pth")
    torch.save(best_checkpoint, best_path)
    logging.info("Best teacher model saved: %s", best_path)

    # Save experiment summary
    save_experiment_summary(config, training_history, test_results, "Teacher")

    logging.info(f"Teacher training completed. Best accuracy: {best_acc:.4f}")

    writer.close()

    return {
        "model": teacher_model,
        "best_accuracy": best_acc,
        "training_history": training_history,
        "test_results": test_results,
        "config": config,
        "experiment_name": experiment_name,
        "teacher_path": best_path,
    }


def get_teacher_path(model_name, dataset_name="default"):
    """Get the path to the trained teacher model"""
    return f"checkpoints/{dataset_name}/teachers/{model_name}_teacher_best.pth"


def create_teacher_config(**kwargs):
    """
    Create a teacher configuration dictionary with custom parameters

    Args:
        **kwargs: Configuration parameters to override defaults

    Returns:
        dict: Configuration dictionary
    """
    config = TEACHER_CONFIG.copy()
    config.update(kwargs)
    return config


def main():
    """Example of training a teacher model"""
    # Create teacher configuration
    teacher_config = create_teacher_config(
        model="ResNet50",  # Teacher model architecture
        data_dir="data",  # Path to your dataset
        batch_size=32,
        epochs=100,  # Train longer for better teacher
        lr=0.1,  # Higher learning rate for teacher
        step_size=30,  # LR schedule
        gamma=0.1,
        device="auto",
    )

    print("Teacher Configuration:")
    for key, value in teacher_config.items():
        print(f"  {key}: {value}")

    # Train the teacher model
    print("\nStarting teacher training...")
    teacher_results = train_teacher(teacher_config)

    print("\nTeacher training completed!")
    print(f"Best accuracy: {teacher_results['best_accuracy']:.4f}")
    print(f"Teacher model saved to: {teacher_results['teacher_path']}")

    # Now you can use this teacher for knowledge distillation
    teacher_path = get_teacher_path("ResNet50")
    print(f"\nTeacher path for distillation: {teacher_path}")

    return teacher_results


if __name__ == "__main__":
    main()
