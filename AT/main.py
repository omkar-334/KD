import logging
import os
import sys
import time

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append("/Users/omkarkabde/Desktop/KD")
from AT.utils import at, at_loss, distillation
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

# Default configuration for AT training
DEFAULT_CONFIG = {
    "model": "MultiBranchNet",
    "teacher_model": None,  # Path to teacher model checkpoint
    "data_dir": "data",
    "batch_size": 32,
    "num_workers": 4,
    "augment": False,
    "epochs": 50,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "step_size": 10,
    "gamma": 0.1,
    "temperature": 4.0,
    "alpha": 0.0,  # Weight for knowledge distillation
    "beta": 1e3,  # Weight for attention transfer
    "device": "auto",
    "print_freq": 100,
    "save_freq": 10,
    "save_dir": "checkpoints",
    "log_dir": "logs",
    "experiment_name": None,
    "resume": None,
}


def load_teacher_model(teacher_path, num_classes, device):
    """Load teacher model from checkpoint"""
    checkpoint = torch.load(teacher_path, map_location=device)

    # Get teacher model architecture from checkpoint config
    teacher_config = checkpoint["config"]
    teacher_model = get_model(teacher_config, num_classes)

    teacher_model.load_state_dict(checkpoint["model_state_dict"])
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    return teacher_model


def extract_attention_maps(model, x, config):
    """Extract attention maps from model features"""
    if config["model"] == "MultiBranchNet":
        logits, features = model(x)
        # Use features from different branches as attention maps
        attention_maps = features
    else:
        # For ResNet models, extract features from intermediate layers
        outputs = model(x)
        # Extract features from intermediate layers (g0, g1, g2, g3)
        attention_maps = outputs[
            5:9
        ]  # middle1_fea, middle2_fea, middle3_fea, final_fea

    return attention_maps


# MAIN Distillation Logic ###
def compute_at_loss(student_model, teacher_model, x, y, config):
    """Compute Attention Transfer loss"""
    criterion = nn.CrossEntropyLoss()

    # Get student predictions and features
    if config["model"] == "MultiBranchNet":
        student_logits, student_features = student_model(x)
        student_pred = student_logits[-1]  # Use final prediction
    else:
        student_outputs = student_model(x)
        student_pred = student_outputs[0]
        student_features = student_outputs[5:9]  # Extract features

    # Get teacher predictions and features
    with torch.no_grad():
        if teacher_model is not None:
            if (
                hasattr(teacher_model, "forward")
                and len(teacher_model.forward(x).shape) == 2
            ):
                # Teacher is a standard model
                teacher_pred = teacher_model(x)
                teacher_features = None  # No intermediate features available
            else:
                # Teacher has intermediate outputs
                teacher_outputs = teacher_model(x)
                teacher_pred = teacher_outputs[0]
                teacher_features = (
                    teacher_outputs[5:9] if len(teacher_outputs) > 5 else None
                )
        else:
            teacher_pred = None
            teacher_features = None

    # Compute classification loss
    classification_loss = criterion(student_pred, y)

    # Compute knowledge distillation loss
    kd_loss = 0
    if teacher_pred is not None and config["alpha"] > 0:
        kd_loss = distillation(
            student_pred, teacher_pred, y, config["temperature"], config["alpha"]
        )

    # Compute attention transfer loss
    at_loss_total = 0
    if teacher_features is not None and config["beta"] > 0:
        # Match attention maps between student and teacher
        min_features = min(len(student_features), len(teacher_features))
        for i in range(min_features):
            at_loss_total += at_loss(student_features[i], teacher_features[i])

    # Total loss
    total_loss = classification_loss + kd_loss + config["beta"] * at_loss_total

    return total_loss, {
        "classification_loss": classification_loss.item(),
        "kd_loss": kd_loss.item() if isinstance(kd_loss, torch.Tensor) else kd_loss,
        "at_loss": at_loss_total.item()
        if isinstance(at_loss_total, torch.Tensor)
        else at_loss_total,
        "total_loss": total_loss.item(),
    }


def train_epoch(student_model, teacher_model, train_loader, optimizer, device, config):
    """Train for one epoch"""
    student_model.train()
    if teacher_model is not None:
        teacher_model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()

    for i, (x, y) in enumerate(tqdm(train_loader, desc="Training")):
        data_time.update(time.time() - end)

        x, y = x.to(device), y.to(device)

        # Compute loss
        loss, loss_dict = compute_at_loss(student_model, teacher_model, x, y, config)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        losses.update(loss_dict["total_loss"], x.size(0))

        # Compute accuracy
        with torch.no_grad():
            if config["model"] == "MultiBranchNet":
                logits, _ = student_model(x)
                pred = logits[-1].argmax(1)
            else:
                outputs = student_model(x)
                pred = outputs[0].argmax(1)

            correct = (pred == y).sum().item()
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

    return losses.avg, top1.avg, loss_dict


def validate(student_model, teacher_model, val_loader, device, config):
    """Validate the model"""
    student_model.eval()
    if teacher_model is not None:
        teacher_model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation"):
            x, y = x.to(device), y.to(device)

            # Compute loss
            loss, loss_dict = compute_at_loss(
                student_model, teacher_model, x, y, config
            )

            # Compute accuracy
            if config["model"] == "MultiBranchNet":
                logits, _ = student_model(x)
                pred = logits[-1].argmax(1)
            else:
                outputs = student_model(x)
                pred = outputs[0].argmax(1)

            correct = (pred == y).sum().item()

            losses.update(loss_dict["total_loss"], x.size(0))
            top1.update(correct / x.size(0) * 100, x.size(0))

    return losses.avg, top1.avg


def train_at(
    config=None,
    train_loader=None,
    val_loader=None,
    test_loader=None,
    dataset_name="default",
):
    """
    Train a model using Attention Transfer knowledge distillation

    Args:
        config (dict): Training configuration. If None, uses DEFAULT_CONFIG
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (optional)
        dataset_name (str): Name of the dataset for organizing outputs

    Returns:
        dict: Training results including model, best accuracy, and metrics
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Setup device
    device = get_device(config["device"])

    # Setup logging
    experiment_name = setup_logging(config, "AT", dataset_name)
    config["experiment_name"] = experiment_name

    # Setup tensorboard
    writer = setup_tensorboard(config, "AT")

    # Prepare data if not provided
    if train_loader is None or val_loader is None:
        logging.info("Preparing data loaders...")
        train_loader, val_loader, test_loader = prepare_data_loaders(config)

    num_classes = len(train_loader.dataset.dataset.classes)
    config["num_classes"] = num_classes

    # Create student model
    logging.info(f"Creating {config['model']} student model...")
    student_model = get_model(config, num_classes)
    student_model = student_model.to(device)

    # Load teacher model
    from teacher import get_teacher_path

    teacher_path = get_teacher_path(config["model"], dataset_name)
    logging.info("Loading teacher model from %s", teacher_path)
    teacher_model = load_teacher_model(teacher_path, num_classes, device)
    logging.info("Teacher model loaded successfully")

    # Count parameters
    total_params, trainable_params = count_parameters(student_model)
    logging.info(f"Student model parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Setup optimizer and scheduler
    optimizer = create_optimizer(student_model, config)
    scheduler = create_scheduler(optimizer, config)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_acc = 0.0

    if config["resume"]:
        logging.info(f"Resuming from checkpoint: {config['resume']}")
        checkpoint = torch.load(config["resume"], map_location=device)
        student_model.load_state_dict(checkpoint["model_state_dict"])
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
    logging.info("Starting training...")
    training_history = []

    for epoch in range(start_epoch, config["epochs"]):
        logging.info(f"Epoch {epoch}/{config['epochs'] - 1}")

        # Train
        train_loss, train_acc, loss_dict = train_epoch(
            student_model, teacher_model, train_loader, optimizer, device, config
        )

        # Validate
        val_loss, val_acc = validate(
            student_model, teacher_model, val_loader, device, config
        )

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
            **loss_dict,
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

        # Log detailed losses
        for key, value in loss_dict.items():
            log_metrics(writer, {f"Loss/{key}": value}, epoch)

        # Save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        if epoch % config["save_freq"] == 0 or is_best:
            save_checkpoint(student_model, optimizer, epoch, best_acc, config, is_best)

    # Final evaluation on test set
    test_results = None
    if test_loader is not None:
        logging.info("Evaluating on test set...")
        test_result = evaluate_model(student_model, test_loader, device, config)
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

    # Save final model
    final_checkpoint = {
        "epoch": config["epochs"],
        "model_state_dict": student_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
        "config": config,
    }
    final_path = os.path.join(config["save_dir"], f"{experiment_name}_final.pth")
    torch.save(final_checkpoint, final_path)
    logging.info("Final model saved: %s", final_path)

    # Save experiment summary
    save_experiment_summary(config, training_history, test_results, "AT")

    logging.info(f"Experiment completed. Best accuracy: {best_acc:.4f}")

    writer.close()

    return {
        "model": student_model,
        "teacher_model": teacher_model,
        "best_accuracy": best_acc,
        "training_history": training_history,
        "test_results": test_results,
        "config": config,
        "experiment_name": experiment_name,
    }


def visualize_attention_maps(model, x, config, layer_idx=0):
    """
    Visualize attention maps for a given input

    Args:
        model: Trained model
        x: Input tensor
        config: Model configuration
        layer_idx: Which layer to visualize (0-based index)

    Returns:
        torch.Tensor: Attention map
    """
    model.eval()
    with torch.no_grad():
        if config["model"] == "MultiBranchNet":
            logits, features = model(x)
            attention_map = at(features[layer_idx])
        else:
            outputs = model(x)
            features = outputs[5:9]  # Extract features
            attention_map = at(features[layer_idx])

    return attention_map


def create_at_config(**kwargs):
    """
    Create an AT configuration dictionary with custom parameters

    Args:
        **kwargs: Configuration parameters to override defaults

    Returns:
        dict: Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    config.update(kwargs)
    return config
