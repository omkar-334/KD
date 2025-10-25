# -------------------------------
# KD Methods
# -------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from tqdm import tqdm


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


def get_resnet_teacher(num_classes=5, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_teacher(model, train_loader, val_loader, device, epochs=10):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    for _ in tqdm(range(epochs)):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
    torch.save(model.state_dict(), "models/teacher.pth")
    return model


def normal_kd(student, teacher, train_loader, val_loader, device, epochs=10):
    """
    Train a student model using Knowledge Distillation from a teacher.

    Args:
        student: student model (MultiBranchNet)
        teacher: teacher model
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        device: "cuda" or "cpu"
        epochs: number of epochs

    Returns:
        Trained student model
    """
    student.to(device)
    teacher.to(device).eval()
    kd_loss = DistillationLoss()
    optimizer = optim.Adam(student.parameters(), lr=1e-3)

    for epoch in tqdm(range(1, epochs + 1)):
        student.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Teacher logits
            with torch.no_grad():
                tlogits = teacher(x)

            # Student logits (last branch only)
            slogits, _ = student(x)
            loss = kd_loss(slogits[-1], tlogits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        student.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = student(x)
                preds = logits[-1].argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total
        print(f"[Normal KD] Epoch {epoch}: Loss = {avg_loss:.4f}, Acc = {acc:.4f}")

    # Save model
    torch.save(student.state_dict(), "models/student.pth")
    return student


if __name__ == "__main__":
    import os

    from model import MultiBranchNet, evaluate, prepare_dataloaders

    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []

    # Standard and augmented loaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        "data", batch_size=32, augment=False, num_workers=4
    )
    train_loader_aug, val_loader, test_loader = prepare_dataloaders(
        "data", batch_size=32, augment=True, num_workers=4
    )

    NUM_CLASSES = len(train_loader.dataset.dataset.classes)

    # ----------------------------
    # Method 1: Normal KD (unchanged)
    # ----------------------------
    teacher = train_teacher(
        get_resnet_teacher(num_classes=NUM_CLASSES),
        train_loader,
        val_loader,
        device,
        epochs=20,
    )
    s1 = normal_kd(
        MultiBranchNet(num_classes=NUM_CLASSES),
        teacher,
        train_loader,
        val_loader,
        device,
        epochs=20,
    )

    acc1 = evaluate(s1, val_loader, device)
    size1 = sum(p.numel() for p in s1.parameters())
    results["Normal KD"] = {"acc": acc1, "size": size1}
    print(acc1, size1)
