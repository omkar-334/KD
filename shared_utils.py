"""
Shared utilities for knowledge distillation methods
This module provides common functions used across BYOT, AT, and DML implementations
"""

import json
import logging
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append("/Users/omkarkabde/Desktop/KD")
import torchvision.models as models

from model import MultiBranchNet, prepare_dataloaders


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def setup_logging(config, method_name="Training", dataset_name="default"):
    """Setup logging configuration"""
    if config["experiment_name"] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config["experiment_name"] = f"{config['model']}_{method_name}_{timestamp}"

    # Create dataset and method-specific directories
    dataset_save_dir = os.path.join(config["save_dir"], dataset_name)
    dataset_log_dir = os.path.join(config["log_dir"], dataset_name)
    method_save_dir = os.path.join(dataset_save_dir, method_name)
    method_log_dir = os.path.join(dataset_log_dir, method_name)

    os.makedirs(method_save_dir, exist_ok=True)
    os.makedirs(method_log_dir, exist_ok=True)

    # Update config with dataset and method-specific directories
    config["save_dir"] = method_save_dir
    config["log_dir"] = method_log_dir

    # Setup logging
    log_file = os.path.join(method_log_dir, f"{config['experiment_name']}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return config["experiment_name"]


def get_model(config, num_classes):
    """Get model based on architecture choice"""
    if config["model"] == "MultiBranchNet":
        model = MultiBranchNet(num_classes=num_classes)
    elif config["model"] == "ResNet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif config["model"] == "ResNet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif config["model"] == "ResNet32":
        from DML.resnet import resnet32

        model = resnet32()
        # Update final layer for correct number of classes
        model.linear = nn.Linear(64, num_classes)
    else:
        raise ValueError(f"Unknown model: {config['model']}")

    return model


def load_model(checkpoint_path, device="auto"):
    """
    Load a trained model from checkpoint

    Args:
        checkpoint_path (str): Path to checkpoint file
        device (str): Device to load model on

    Returns:
        dict: Model and metadata from checkpoint
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    # Create model
    model = get_model(config, config.get("num_classes", 5))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    return {
        "model": model,
        "config": config,
        "epoch": checkpoint["epoch"],
        "best_acc": checkpoint["best_acc"],
    }


def save_checkpoint(
    model, optimizer, epoch, best_acc, config, is_best=False, model_idx=None
):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_acc": best_acc,
        "config": config,
    }

    # Add model index to filename if provided (for DML)
    if model_idx is not None:
        checkpoint_path = os.path.join(
            config["save_dir"],
            f"{config['experiment_name']}_model{model_idx}_epoch_{epoch}.pth",
        )
        if is_best:
            best_path = os.path.join(
                config["save_dir"],
                f"{config['experiment_name']}_model{model_idx}_best.pth",
            )
    else:
        checkpoint_path = os.path.join(
            config["save_dir"], f"{config['experiment_name']}_epoch_{epoch}.pth"
        )
        if is_best:
            best_path = os.path.join(
                config["save_dir"], f"{config['experiment_name']}_best.pth"
            )

    torch.save(checkpoint, checkpoint_path)

    if is_best:
        torch.save(checkpoint, best_path)
        logging.info("New best model saved: %s", best_path)

    logging.info("Checkpoint saved: %s", checkpoint_path)


def evaluate_model(model, data_loader, device="auto", config=None):
    """
    Evaluate a model on a dataset

    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device (str): Device to use
        config (dict): Model configuration (for model type)

    Returns:
        dict: Evaluation results
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model.eval()
    model = model.to(device)

    losses = AverageMeter()
    top1 = AverageMeter()

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Evaluation"):
            x, y = x.to(device), y.to(device)

            # Forward pass
            if config and config["model"] == "MultiBranchNet":
                logits, _ = model(x)
                pred = logits[-1]
            else:
                pred = model(x)

            loss = criterion(pred, y)
            correct = (pred.argmax(1) == y).sum().item()

            losses.update(loss.item(), x.size(0))
            top1.update(correct / x.size(0) * 100, x.size(0))

    return {
        "loss": losses.avg,
        "accuracy": top1.avg,
        "total_samples": len(data_loader.dataset),
    }


def create_config(**kwargs):
    """
    Create a configuration dictionary with custom parameters

    Args:
        **kwargs: Configuration parameters to override defaults

    Returns:
        dict: Configuration dictionary
    """
    # Base configuration
    base_config = {
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
        "device": "auto",
        "print_freq": 100,
        "save_freq": 10,
        "save_dir": "checkpoints",
        "log_dir": "logs",
        "experiment_name": None,
        "resume": None,
    }

    base_config.update(kwargs)
    return base_config


def setup_tensorboard(config, method_name="Training"):
    """Setup TensorBoard logging"""
    writer = SummaryWriter(os.path.join(config["log_dir"], config["experiment_name"]))
    return writer


def log_metrics(writer, metrics, epoch, prefix=""):
    """Log metrics to TensorBoard"""
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, epoch)


def save_experiment_summary(
    config, training_history, test_results=None, method_name="Training"
):
    """Save experiment summary to JSON file"""
    summary = {
        "experiment_name": config["experiment_name"],
        "method": method_name,
        "model": config["model"],
        "best_accuracy": max([epoch["val_acc"] for epoch in training_history])
        if training_history
        else 0,
        "total_epochs": config["epochs"],
        "config": config,
        "training_history": training_history,
        "test_results": test_results,
    }

    summary_path = os.path.join(
        config["log_dir"], f"{config['experiment_name']}_summary.json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("Summary saved: %s", summary_path)
    return summary_path


def get_device(device_config):
    """Get device based on configuration"""
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def create_optimizer(model, config):
    """Create optimizer based on configuration"""
    return optim.SGD(
        model.parameters(),
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )


def create_scheduler(optimizer, config):
    """Create learning rate scheduler based on configuration"""
    return optim.lr_scheduler.StepLR(
        optimizer, step_size=config["step_size"], gamma=config["gamma"]
    )


def prepare_data_loaders(config):
    """Prepare data loaders based on configuration"""
    return prepare_dataloaders(
        config["data_dir"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        augment=True,
    )


def print_training_info(
    config, total_params, trainable_params, num_classes, num_train, num_val
):
    """Print training information"""
    logging.info("Starting experiment: %s", config["experiment_name"])
    logging.info("Using device: %s", get_device(config["device"]))
    logging.info("Configuration: %s", config)
    logging.info("Number of classes: %s", num_classes)
    logging.info("Training samples: %s", num_train)
    logging.info("Validation samples: %s", num_val)
    logging.info("Total parameters: %s", f"{total_params:,}")
    logging.info("Trainable parameters: %s", f"{trainable_params:,}")
