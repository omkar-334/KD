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

# Default configuration for BYOT training
DEFAULT_CONFIG = {
    "model": "MultiBranchNet",
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
    "temperature": 3.0,
    "alpha": 0.1,
    "beta": 1e-6,
    "device": "auto",
    "print_freq": 100,
    "save_freq": 10,
    "save_dir": "checkpoints",
    "log_dir": "logs",
    "experiment_name": None,
    "resume": None,
}


def kd_loss_function(student_output, teacher_output, temperature):
    """Compute knowledge distillation loss"""
    student_output = student_output / temperature
    student_log_softmax = torch.log_softmax(student_output, dim=1)
    teacher_softmax = torch.softmax(teacher_output / temperature, dim=1)
    loss_kd = -torch.mean(torch.sum(student_log_softmax * teacher_softmax, dim=1))
    return loss_kd * (temperature**2)


def feature_loss_function(student_feat, teacher_feat):
    """Compute feature matching loss"""
    loss = (student_feat - teacher_feat) ** 2 * (
        (student_feat > 0) | (teacher_feat > 0)
    ).float()
    return torch.abs(loss).sum()


# MAIN Distillation Logic ###
def compute_byot_loss(student_model, x, y, config, teacher_model=None):
    """Compute BYOT loss for different model architectures"""
    criterion = nn.CrossEntropyLoss()

    if config["model"] == "MultiBranchNet":
        # MultiBranchNet returns (logits, features)
        logits, features = student_model(x)

        # Final prediction loss
        final_loss = criterion(logits[-1], y)

        # Intermediate losses
        intermediate_losses = sum(
            criterion(logits[i], y) for i in range(len(logits) - 1)
        )

        # Knowledge distillation losses
        kd_losses = 0
        feature_losses = 0

        # Use final output as teacher for intermediate outputs
        teacher_output = logits[-1].detach()
        for i in range(len(logits) - 1):
            kd_loss = kd_loss_function(logits[i], teacher_output, config["temperature"])
            kd_losses += kd_loss

            # Feature matching loss
            if i < len(features) - 1:
                feat_loss = feature_loss_function(features[i], features[-1].detach())
                feature_losses += feat_loss

        total_loss = (
            (1 - config["alpha"]) * (final_loss + intermediate_losses)
            + config["alpha"] * kd_losses
            + config["beta"] * feature_losses
        )

        return total_loss, {
            "final_loss": final_loss.item(),
            "intermediate_loss": intermediate_losses.item(),
            "kd_loss": kd_losses.item(),
            "feature_loss": feature_losses.item(),
            "total_loss": total_loss.item(),
        }

    # ResNet models
    # ResNet models return (output, middle_output1, middle_output2, middle_output3, final_fea, middle1_fea, middle2_fea, middle3_fea)
    outputs = student_model(x)
    final_output = outputs[0]
    middle_outputs = outputs[1:4]
    final_feat = outputs[4]
    middle_feats = outputs[5:8]

    # Final prediction loss
    final_loss = criterion(final_output, y)

    # Intermediate losses
    intermediate_losses = sum(
        criterion(middle_outputs[i], y) for i in range(len(middle_outputs))
    )

    # Knowledge distillation losses
    kd_losses = 0
    feature_losses = 0

    teacher_output = final_output.detach()
    for i, middle_output in enumerate(middle_outputs):
        kd_loss = kd_loss_function(middle_output, teacher_output, config["temperature"])
        kd_losses += kd_loss

        # Feature matching loss
        if i < len(middle_feats):
            feat_loss = feature_loss_function(middle_feats[i], final_feat.detach())
            feature_losses += feat_loss

    total_loss = (
        (1 - config["alpha"]) * (final_loss + intermediate_losses)
        + config["alpha"] * kd_losses
        + config["beta"] * feature_losses
    )

    return total_loss, {
        "final_loss": final_loss.item(),
        "intermediate_loss": intermediate_losses.item(),
        "kd_loss": kd_losses.item(),
        "feature_loss": feature_losses.item(),
        "total_loss": total_loss.item(),
    }


def train_epoch(student_model, train_loader, optimizer, device, config, teacher_model):
    """Train student model for one epoch"""
    student_model.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()

    for i, (x, y) in enumerate(tqdm(train_loader, desc="Training")):
        data_time.update(time.time() - end)

        x, y = x.to(device), y.to(device)

        # Compute loss
        loss, loss_dict = compute_byot_loss(student_model, x, y, config, teacher_model)

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


def validate(student_model, val_loader, device, config, teacher_model):
    """Validate the student model"""
    student_model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation"):
            x, y = x.to(device), y.to(device)

            # Compute loss
            loss, loss_dict = compute_byot_loss(
                student_model, x, y, config, teacher_model
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


def train_byot(
    config=None,
    train_loader=None,
    val_loader=None,
    test_loader=None,
    dataset_name="default",
):
    """
    Train a model using BYOT knowledge distillation

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
    experiment_name = setup_logging(config, "BYOT", dataset_name)
    config["experiment_name"] = experiment_name

    # Setup tensorboard
    writer = setup_tensorboard(config, "BYOT")

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
    teacher_checkpoint = torch.load(teacher_path, map_location=device)
    teacher_config = teacher_checkpoint["config"]
    teacher_model = get_model(teacher_config, num_classes)
    teacher_model.load_state_dict(teacher_checkpoint["model_state_dict"])
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    logging.info("Teacher model loaded successfully")

    # Count parameters
    total_params, trainable_params = count_parameters(student_model)
    logging.info(f"Total parameters: {total_params:,}")
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
            student_model, train_loader, optimizer, device, config, teacher_model
        )

        # Validate
        val_loss, val_acc = validate(
            student_model, val_loader, device, config, teacher_model
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
    save_experiment_summary(config, training_history, test_results, "BYOT")

    logging.info(f"Experiment completed. Best accuracy: {best_acc:.4f}")

    writer.close()

    return {
        "model": student_model,
        "best_accuracy": best_acc,
        "training_history": training_history,
        "test_results": test_results,
        "config": config,
        "experiment_name": experiment_name,
    }


def create_byot_config(**kwargs):
    """
    Create a BYOT configuration dictionary with custom parameters

    Args:
        **kwargs: Configuration parameters to override defaults

    Returns:
        dict: Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    config.update(kwargs)
    return config
