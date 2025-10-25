import logging
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append("/Users/omkarkabde/Desktop/KD")
from DML.resnet import resnet20, resnet32, resnet44, resnet56, resnet110
from shared_utils import (
    AverageMeter,
    count_parameters,
    create_optimizer,
    create_scheduler,
    evaluate_model,
    get_device,
    log_metrics,
    prepare_data_loaders,
    print_training_info,
    save_checkpoint,
    save_experiment_summary,
    setup_logging,
    setup_tensorboard,
)

# Default configuration for DML training
DEFAULT_CONFIG = {
    "model": "ResNet32",
    "data_dir": "data",
    "batch_size": 32,
    "num_workers": 4,
    "augment": False,
    "epochs": 200,
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "step_size": 60,
    "gamma": 0.1,
    "model_num": 2,  # Number of models for DML
    "device": "auto",
    "print_freq": 10,
    "save_freq": 10,
    "save_dir": "checkpoints",
    "log_dir": "logs",
    "experiment_name": None,
    "resume": None,
}


def get_dml_model(config, num_classes):
    """Get DML model based on architecture choice"""
    if config["model"] == "ResNet20":
        model = resnet20()
    elif config["model"] == "ResNet32":
        model = resnet32()
    elif config["model"] == "ResNet44":
        model = resnet44()
    elif config["model"] == "ResNet56":
        model = resnet56()
    elif config["model"] == "ResNet110":
        model = resnet110()
    else:
        raise ValueError(f"Unknown DML model: {config['model']}")

    # Update final layer for correct number of classes
    model.linear = nn.Linear(64, num_classes)
    return model


# MAIN Distillation Logic ###
def compute_dml_loss(models, x, y, model_idx):
    """
    Compute DML loss for a specific model

    Args:
        models: List of all models
        x: Input batch
        y: Target labels
        model_idx: Index of the current model

    Returns:
        tuple: (total_loss, loss_dict)
    """
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction="batchmean")

    # Get current model output
    current_output = models[model_idx](x)
    ce_loss = criterion_ce(current_output, y)

    # Compute KL divergence loss with other models
    kl_loss = 0
    for j, other_model in enumerate(models):
        if j != model_idx:
            with torch.no_grad():
                other_output = other_model(x)
            kl_loss += criterion_kl(
                F.log_softmax(current_output, dim=1), F.softmax(other_output, dim=1)
            )

    # Average KL loss over other models
    if len(models) > 1:
        kl_loss = kl_loss / (len(models) - 1)

    total_loss = ce_loss + kl_loss

    return total_loss, {
        "ce_loss": ce_loss.item(),
        "kl_loss": kl_loss.item(),
        "total_loss": total_loss.item(),
    }


def train_epoch_dml(models, optimizers, train_loader, device, config):
    """Train DML models for one epoch"""
    for model in models:
        model.train()

    losses = [AverageMeter() for _ in range(len(models))]
    accs = [AverageMeter() for _ in range(len(models))]
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()

    for i, (x, y) in enumerate(tqdm(train_loader, desc="Training")):
        data_time.update(time.time() - end)
        x, y = x.to(device), y.to(device)

        # Train each model
        for model_idx in range(len(models)):
            # Compute loss
            loss, loss_dict = compute_dml_loss(models, x, y, model_idx)

            # Backward pass
            optimizers[model_idx].zero_grad()
            loss.backward()
            optimizers[model_idx].step()

            # Update metrics
            losses[model_idx].update(loss_dict["total_loss"], x.size(0))

            # Compute accuracy
            with torch.no_grad():
                output = models[model_idx](x)
                pred = output.argmax(1)
                correct = (pred == y).sum().item()
                accs[model_idx].update(correct / x.size(0) * 100, x.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config["print_freq"] == 0:
            logging.info(
                f"Batch [{i}/{len(train_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"Model1 Loss: {losses[0].val:.4f} ({losses[0].avg:.4f})\t"
                f"Model1 Acc: {accs[0].val:.3f} ({accs[0].avg:.3f})"
            )

    return losses, accs


def validate_dml(models, val_loader, device, config):
    """Validate DML models"""
    for model in models:
        model.eval()

    losses = [AverageMeter() for _ in range(len(models))]
    accs = [AverageMeter() for _ in range(len(models))]

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation"):
            x, y = x.to(device), y.to(device)

            # Validate each model
            for model_idx in range(len(models)):
                # Compute loss
                loss, loss_dict = compute_dml_loss(models, x, y, model_idx)

                # Compute accuracy
                output = models[model_idx](x)
                pred = output.argmax(1)
                correct = (pred == y).sum().item()

                losses[model_idx].update(loss_dict["total_loss"], x.size(0))
                accs[model_idx].update(correct / x.size(0) * 100, x.size(0))

    return losses, accs


def train_dml(
    config=None,
    train_loader=None,
    val_loader=None,
    test_loader=None,
    dataset_name="default",
):
    """
    Train models using Deep Mutual Learning (DML)

    Args:
        config (dict): Training configuration. If None, uses DEFAULT_CONFIG
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader (optional)
        dataset_name (str): Name of the dataset for organizing outputs

    Returns:
        dict: Training results including models, best accuracy, and metrics
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    # Setup device
    device = get_device(config["device"])

    # Setup logging
    experiment_name = setup_logging(config, "DML", dataset_name)
    config["experiment_name"] = experiment_name

    # Setup tensorboard
    writer = setup_tensorboard(config, "DML")

    # Prepare data if not provided
    if train_loader is None or val_loader is None:
        logging.info("Preparing data loaders...")
        train_loader, val_loader, test_loader = prepare_data_loaders(config)

    num_classes = len(train_loader.dataset.dataset.classes)
    config["num_classes"] = num_classes

    # Create models
    logging.info(f"Creating {config['model_num']} {config['model']} models...")
    models = []
    for i in range(config["model_num"]):
        model = get_dml_model(config, num_classes)
        model = model.to(device)
        models.append(model)

    # Count parameters
    total_params, trainable_params = count_parameters(models[0])
    logging.info(f"Model parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Setup optimizers and schedulers
    optimizers = []
    schedulers = []
    for model in models:
        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(optimizer, config)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_accs = [0.0] * config["model_num"]

    if config["resume"]:
        logging.info(f"Resuming from checkpoint: {config['resume']}")
        # Note: DML checkpoint loading would need special handling for multiple models
        # This is a simplified version
        logging.warning("DML checkpoint resuming not fully implemented yet")

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
    logging.info("Starting DML training...")
    training_history = []

    for epoch in range(start_epoch, config["epochs"]):
        logging.info(f"Epoch {epoch}/{config['epochs'] - 1}")

        # Update learning rate
        for scheduler in schedulers:
            scheduler.step()
        current_lr = optimizers[0].param_groups[0]["lr"]

        # Train
        train_losses, train_accs = train_epoch_dml(
            models, optimizers, train_loader, device, config
        )

        # Validate
        val_losses, val_accs = validate_dml(models, val_loader, device, config)

        # Log metrics
        logging.info(f"Epoch {epoch}: LR: {current_lr:.6f}")

        for i in range(config["model_num"]):
            logging.info(
                f"Model {i + 1}: "
                f"Train Loss: {train_losses[i].avg:.4f}, Train Acc: {train_accs[i].avg:.4f}, "
                f"Val Loss: {val_losses[i].avg:.4f}, Val Acc: {val_accs[i].avg:.4f}"
            )

        # Store epoch results
        epoch_results = {"epoch": epoch, "lr": current_lr, "models": []}

        for i in range(config["model_num"]):
            epoch_results["models"].append({
                "model_idx": i,
                "train_loss": train_losses[i].avg,
                "train_acc": train_accs[i].avg,
                "val_loss": val_losses[i].avg,
                "val_acc": val_accs[i].avg,
            })

        training_history.append(epoch_results)

        # Tensorboard logging
        log_metrics(writer, {"Learning_Rate": current_lr}, epoch)
        for i in range(config["model_num"]):
            log_metrics(
                writer,
                {
                    f"Model{i + 1}/Train_Loss": train_losses[i].avg,
                    f"Model{i + 1}/Train_Accuracy": train_accs[i].avg,
                    f"Model{i + 1}/Val_Loss": val_losses[i].avg,
                    f"Model{i + 1}/Val_Accuracy": val_accs[i].avg,
                },
                epoch,
            )

        # Save checkpoints
        for i in range(config["model_num"]):
            is_best = val_accs[i].avg > best_accs[i]
            best_accs[i] = max(val_accs[i].avg, best_accs[i])

            if epoch % config["save_freq"] == 0 or is_best:
                save_checkpoint(
                    models[i], optimizers[i], epoch, best_accs[i], config, is_best, i
                )

    # Final evaluation on test set
    test_results = None
    if test_loader is not None:
        logging.info("Evaluating on test set...")
        test_losses = []
        test_accs = []

        for i, model in enumerate(models):
            test_result = evaluate_model(model, test_loader, device, config)
            test_losses.append(test_result["loss"])
            test_accs.append(test_result["accuracy"])
            logging.info(
                f"Model {i + 1} Test Loss: {test_result['loss']:.4f}, Test Acc: {test_result['accuracy']:.4f}"
            )

        test_results = {
            "test_losses": test_losses,
            "test_accs": test_accs,
            "avg_test_acc": sum(test_accs) / len(test_accs),
        }

    # Save final models
    for i, model in enumerate(models):
        final_checkpoint = {
            "epoch": config["epochs"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizers[i].state_dict(),
            "best_acc": best_accs[i],
            "config": config,
        }
        final_path = os.path.join(
            config["save_dir"], f"{experiment_name}_model{i + 1}_final.pth"
        )
        torch.save(final_checkpoint, final_path)
        logging.info("Final model %d saved: %s", i + 1, final_path)

    # Save experiment summary
    save_experiment_summary(config, training_history, test_results, "DML")

    logging.info(
        f"DML experiment completed. Average best accuracy: {sum(best_accs) / len(best_accs):.4f}"
    )

    writer.close()

    return {
        "models": models,
        "best_accuracies": best_accs,
        "avg_best_accuracy": sum(best_accs) / len(best_accs),
        "training_history": training_history,
        "test_results": test_results,
        "config": config,
        "experiment_name": experiment_name,
    }


def load_dml_models(checkpoint_paths, device="auto"):
    """
    Load DML models from checkpoints

    Args:
        checkpoint_paths: List of checkpoint file paths
        device (str): Device to load models on

    Returns:
        dict: Models and metadata from checkpoints
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    models = []
    configs = []

    for i, checkpoint_path in enumerate(checkpoint_paths):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint["config"]

        # Create model
        model = get_dml_model(config, config.get("num_classes", 100))
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        models.append(model)
        configs.append(config)

    return {
        "models": models,
        "configs": configs,
        "device": device,
    }


def evaluate_dml_models(models, data_loader, device="auto", config=None):
    """
    Evaluate DML models on a dataset

    Args:
        models: List of trained models
        data_loader: Data loader for evaluation
        device (str): Device to use
        config (dict): Model configuration

    Returns:
        dict: Evaluation results for each model
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    results = []

    for i, model in enumerate(models):
        model.eval()
        model = model.to(device)

        result = evaluate_model(model, data_loader, device, config)
        result["model_idx"] = i
        results.append(result)

    # Calculate ensemble results
    if len(models) > 1:
        ensemble_acc = evaluate_ensemble(models, data_loader, device, config)
        results.append({
            "model_idx": "ensemble",
            "loss": ensemble_acc["loss"],
            "accuracy": ensemble_acc["accuracy"],
            "total_samples": ensemble_acc["total_samples"],
        })

    return results


def evaluate_ensemble(models, data_loader, device="auto", config=None):
    """
    Evaluate ensemble of DML models

    Args:
        models: List of trained models
        data_loader: Data loader for evaluation
        device (str): Device to use
        config (dict): Model configuration

    Returns:
        dict: Ensemble evaluation results
    """
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    for model in models:
        model.eval()
        model = model.to(device)

    losses = AverageMeter()
    top1 = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in tqdm(data_loader, desc="Ensemble Evaluation"):
            x, y = x.to(device), y.to(device)

            # Get predictions from all models
            predictions = []
            for model in models:
                pred = model(x)
                predictions.append(pred)

            # Average predictions
            ensemble_pred = torch.stack(predictions).mean(dim=0)
            loss = criterion(ensemble_pred, y)
            correct = (ensemble_pred.argmax(1) == y).sum().item()

            losses.update(loss.item(), x.size(0))
            top1.update(correct / x.size(0) * 100, x.size(0))

    return {
        "loss": losses.avg,
        "accuracy": top1.avg,
        "total_samples": len(data_loader.dataset),
    }


def create_dml_config(**kwargs):
    """
    Create a DML configuration dictionary with custom parameters

    Args:
        **kwargs: Configuration parameters to override defaults

    Returns:
        dict: Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    config.update(kwargs)
    return config
