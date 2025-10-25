import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


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

    for epoch in tqdm(range(1, num_epochs + 1)):
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


if __name__ == "__main__":
    import os

    from model import MultiBranchNet, evaluate, prepare_dataloaders

    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader = prepare_dataloaders(
        "data", batch_size=32, augment=True, num_workers=4
    )
    NUM_CLASSES = len(train_loader.dataset.dataset.classes)
    model = MultiBranchNet(num_classes=NUM_CLASSES)

    s3 = train(
        MultiBranchNet(num_classes=NUM_CLASSES),
        train_loader,
        val_loader,
        device,
        qat=False,
        num_epochs=20,
        method_name="MTSD",
    )

    acc3 = evaluate(s3, val_loader, device)
    size3 = sum(p.numel() for p in s3.parameters())
    print(acc3, size3)
