import torch
import torch.nn as nn

# -------------------------------
# Simple MTSD model
# -------------------------------


# ========== ECA and Backbone ==========
class ECALayer(nn.Module):
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(1, 2)
        y = self.conv(y).transpose(1, 2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Branch(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


class MultiBranchNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=5):
        super().__init__()
        self.blocks = nn.ModuleList([
            BottleneckBlock(input_channels, 64),
            BottleneckBlock(64, 128),
            BottleneckBlock(128, 256),
            BottleneckBlock(256, 512),
        ])
        self.att = nn.ModuleList([
            ECALayer(64),
            ECALayer(128),
            ECALayer(256),
            ECALayer(512),
        ])
        self.classifiers = nn.ModuleList([
            Branch(64, num_classes),
            Branch(128, num_classes),
            Branch(256, num_classes),
            Branch(512, num_classes),
        ])

    def forward(self, x):
        features, logits = [], []
        for i in range(4):
            x = self.blocks[i](x)
            x = self.att[i](x)
            features.append(x)
            logits.append(self.classifiers[i](x))
        return logits, features


def evaluate(model, loader, device):
    """
    Evaluate a MultiBranchNet or similar model.

    Args:
        model: PyTorch model
        loader: DataLoader for evaluation
        device: "cuda" or "cpu"

    Returns:
        Accuracy (float)
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)  # unpack logits and features
            preds = logits[-1].argmax(1)  # use last branch for final prediction
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def prepare_dataloaders(
    path, batch_size=32, num_workers=2, val_split=0.2, test_split=0.1, augment=False
):
    """
    Prepares train, validation, and test dataloaders from an ImageFolder dataset.
    Supports optional data augmentation for the training set.

    Args:
        path (str): Root directory path to the dataset.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.
        val_split (float): Fraction of data to use for validation.
        test_split (float): Fraction of data to use for testing.
        augment (bool): If True, apply augmentation to training dataset.

    Returns:
        tuple: (train_loader, val_loader, test_loader or None)
    """
    # Base transform (resize + normalize)
    base_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Augmented transform for training only
    aug_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load full dataset (with base transform first)
    full_dataset = datasets.ImageFolder(root=path, transform=base_transform)

    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size

    if test_split > 0:
        train_set, val_set, test_set = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )
    else:
        train_set, val_set = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        test_loader = None

    # Apply augmentation only to training subset if requested
    if augment:
        train_set.dataset.transform = aug_transform

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
