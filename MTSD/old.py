import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

# Global lists to store metrics for plotting
training_metrics = {
    "Normal KD": {"loss": [], "acc": []},
    "Multi-Teacher KD": {"loss": [], "acc": []},
    "MTSD + Aug": {"loss": [], "acc": []},
    "MTSD + Aug + QAT": {"loss": [], "acc": []},
}


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


def compute_asm(feature_map, target_size=None, max_hw=400):
    if target_size is not None:
        feature_map = F.interpolate(
            feature_map, size=target_size, mode="bilinear", align_corners=False
        )
    B, C, H, W = feature_map.size()
    if max_hw < H * W:
        feature_map = F.adaptive_avg_pool2d(
            feature_map, (int(max_hw**0.5), int(max_hw**0.5))
        )
    F_T = feature_map.view(B, C, -1)
    sim = torch.bmm(F_T.transpose(1, 2), F_T)
    norm = torch.norm(sim, dim=(1, 2), keepdim=True) + 1e-6
    return sim / norm


def compute_total_loss(preds, feats, labels, alpha=3, beta=0.3, gamma=3000, delta=None):
    ce = nn.CrossEntropyLoss()
    l1 = ce(preds[3], labels)
    l2 = sum(ce(preds[i], labels) for i in range(3))
    kl_loss, asm_loss = 0, 0
    target_size = feats[3].shape[2:]
    for i in range(3):
        pi = F.log_softmax(preds[i], dim=1)
        for j in range(i + 1, 4):
            pj = F.softmax(preds[j].detach(), dim=1)
            kl = F.kl_div(pi, pj, reduction="batchmean")
            asm_s = compute_asm(feats[i], target_size)
            asm_t = compute_asm(feats[j].detach(), target_size)
            if asm_s.shape != asm_t.shape:
                min_shape = list(map(min, asm_s.shape, asm_t.shape))
                asm_s = asm_s[:, : min_shape[1], : min_shape[2]]
                asm_t = asm_t[:, : min_shape[1], : min_shape[2]]
            asm = F.mse_loss(asm_s, asm_t)
            w = 1.0
            kl_loss += w * kl
            asm_loss += w * asm
    return alpha * l1 + (1 - beta) * l2 + beta * kl_loss + gamma * asm_loss


def train(
    model,
    train_loader,
    val_loader,
    device,
    qat=False,
    num_epochs=50,
    method_name="MTSD",
):
    model.to(device)

    # ---- QAT setup ----
    if qat:
        # Define QAT configuration
        model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
        # Prepare model for QAT
        torch.quantization.prepare_qat(model, inplace=True)
        print("[QAT] Model prepared for Quantization-Aware Training")

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    delta = torch.tensor(0.5, requires_grad=True, device=device)
    delta_opt = optim.Adam([delta], lr=1e-3)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds, feats = model(x)
            loss = compute_total_loss(preds, feats, y, delta=delta)
            loss.backward()
            optimizer.step()
            delta_opt.step()
            total_loss += loss.item()
        scheduler.step()

        val_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds, _ = model(x)
                pred = preds[-1].argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / total
        training_metrics[method_name]["loss"].append(val_loss)
        training_metrics[method_name]["acc"].append(acc)
        print(
            f"[MTSD{'-QAT' if qat else ''}] Epoch {epoch}: Loss = {val_loss:.4f}, Acc = {acc:.4f}"
        )

    # ---- Convert to quantized model after QAT ----
    if qat:
        model.cpu()
        torch.quantization.convert(model, inplace=True)
        print("[QAT] Model converted to quantized version")
    torch.save(model.state_dict(), f"models/mtsd_{method_name}.pth")
    return model


# -------------------------------
# Losses
# -------------------------------
class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        hard_loss = self.ce(student_logits, targets)
        T = self.temperature
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction="batchmean",
        ) * (T * T)
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss


# -------------------------------
# Training helpers
# -------------------------------


def get_resnet_teacher(num_classes=5, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_teacher(model, train_loader, val_loader, device, epochs=10):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
    torch.save(model.state_dict(), "models/teacher.pth")
    return model


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


# -------------------------------
# KD Methods
# -------------------------------
def normal_kd(student, teacher, train_loader, val_loader, device, epochs=10):
    student.to(device)
    teacher.to(device).eval()
    kd_loss = DistillationLoss()
    opt = optim.Adam(student.parameters(), lr=1e-3)

    for epoch in range(epochs):
        student.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                tlogits = teacher(x)
            slogits, _ = student(x)  # unpack logits and features
            loss = kd_loss(slogits[-1], tlogits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        student.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = student(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total

        training_metrics["Normal KD"]["loss"].append(avg_loss)
        training_metrics["Normal KD"]["acc"].append(acc)

        print(f"[Normal KD] Epoch {epoch}: Loss = {avg_loss:.4f}, Acc = {acc:.4f}")

    torch.save(student.state_dict(), "models/student.pth")
    return student


def run_all_experiments():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Standard and augmented loaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        "data4/rice_dataset", batch_size=32, augment=False
    )
    train_loader_aug, val_loader, test_loader = prepare_dataloaders(
        "data4/rice_dataset", batch_size=32, augment=True
    )

    # --- Teachers ---
    results = []
    teacher = train_teacher(
        get_resnet_teacher(num_classes=5), train_loader, val_loader, device, epochs=5
    )

    # ----------------------------
    # Method 1: Normal KD (unchanged)
    # ----------------------------
    s1 = normal_kd(
        MultiBranchNet(num_classes=4),
        teacher,
        train_loader,
        val_loader,
        device,
        epochs=5,
    )
    acc1 = evaluate(s1, val_loader, device)
    size1 = sum(p.numel() for p in s1.parameters())
    results.append(["Normal KD", acc1, size1])

    # ----------------------------
    # Method 2: Multi-Teacher KD (MTSD)
    # ----------------------------
    s2 = train(
        MultiBranchNet(num_classes=4),
        train_loader,
        val_loader,
        device,
        qat=False,
        num_epochs=5,
        method_name="MTSD",
    )
    acc2 = evaluate(s2, val_loader, device)
    size2 = sum(p.numel() for p in s2.parameters())
    results.append(["Multi-Teacher KD", acc2, size2])

    # ----------------------------
    # Method 3: MTSD + Augmentation
    # ----------------------------
    s3 = train(
        MultiBranchNet(num_classes=4),
        train_loader_aug,
        val_loader,
        device,
        qat=False,
        num_epochs=5,
        method_name="MTSD-Aug",
    )
    acc3 = evaluate(s3, val_loader, device)
    size3 = sum(p.numel() for p in s3.parameters())
    results.append(["MTSD + Aug", acc3, size3])

    # ----------------------------
    # Method 4: MTSD + Aug + QAT
    # ----------------------------
    s4 = train(
        MultiBranchNet(num_classes=4),
        train_loader_aug,
        val_loader,
        device,
        qat=True,
        num_epochs=5,
        method_name="MTSD-Aug-QAT",
    )
    acc4 = evaluate(s4, val_loader, "cpu")  # quantized runs on CPU
    size4 = sum(p.numel() for p in s4.parameters())
    results.append(["MTSD + Aug + QAT", acc4, size4])

    # ----------------------------
    # Save Results
    # ----------------------------
    df = pd.DataFrame(results, columns=["Method", "ValAccuracy", "NumParams"])
    df.to_csv("results.csv", index=False)
    print(df)


def plot_metrics(metrics_dict):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for method, vals in metrics_dict.items():
        plt.plot(vals["loss"], label=method)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    for method, vals in metrics_dict.items():
        plt.plot(vals["acc"], label=method)
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_all_experiments()
    plot_metrics(training_metrics)
