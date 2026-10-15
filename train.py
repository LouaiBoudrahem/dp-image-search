"""Train a ResNet-18 on Oxford-IIIT Pets (37 breeds).

Examples:
    python train.py                    # pretrained (fine-tuning), seed 0
    python train.py --seed 1
    python train.py --scratch          # ablation: no pretraining
    python train.py --epochs 2         # quick test run

Saves the best model (by validation accuracy) to models/ and a results line to results.csv.
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch as th
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
NUM_CLASSES = 37


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def get_device():
    if th.cuda.is_available():
        return "cuda"
    if th.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_loaders(data_dir, batch_size, seed, augment=True):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]) if augment else None
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    if train_tf is None:
        train_tf = test_tf

    trainval_aug = datasets.OxfordIIITPet(data_dir, split="trainval", download=True, transform=train_tf)
    trainval_plain = datasets.OxfordIIITPet(data_dir, split="trainval", download=True, transform=test_tf)
    test = datasets.OxfordIIITPet(data_dir, split="test", download=True, transform=test_tf)

    idx = np.random.RandomState(seed).permutation(len(trainval_aug))
    n_val = int(0.15 * len(idx))
    val_idx, train_idx = idx[:n_val].tolist(), idx[n_val:].tolist()

    workers = min(4, os.cpu_count() or 1)
    kw = dict(batch_size=batch_size, num_workers=workers, pin_memory=th.cuda.is_available())
    train_loader = DataLoader(Subset(trainval_aug, train_idx), shuffle=True, **kw)
    val_loader = DataLoader(Subset(trainval_plain, val_idx), shuffle=False, **kw)
    test_loader = DataLoader(test, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


def build_model(pretrained):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # 512 -> 37
    return model


@th.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct1 = correct5 = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        top5 = logits.topk(5, dim=1).indices
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += (top5 == labels[:, None]).any(dim=1).sum().item()
        total += labels.size(0)
    return correct1 / total, correct5 / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scratch", action="store_true", help="train from scratch (no ImageNet weights)")
    p.add_argument("--no-augment", action="store_true", help="ablation: disable data augmentation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None, help="default: 10 pretrained, 30 scratch")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-4 pretrained, 1e-3 scratch")
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    pretrained = not args.scratch
    epochs = args.epochs or (10 if pretrained else 30)
    lr = args.lr or (1e-4 if pretrained else 1e-3)
    device = get_device()
    set_seed(args.seed)

    name = f"resnet18_{'pretrained' if pretrained else 'scratch'}{'_noaug' if args.no_augment else ''}_seed{args.seed}"
    print(f"Run: {name} | device: {device} | epochs: {epochs} | lr: {lr}")

    train_loader, val_loader, test_loader = make_loaders(
        args.data_dir, args.batch_size, args.seed, augment=not args.no_augment)
    model = build_model(pretrained).to(device)

    if pretrained:
        head = list(model.fc.parameters())
        backbone = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
        optimizer = th.optim.Adam([{"params": backbone, "lr": lr}, {"params": head, "lr": lr * 10}])
    else:
        optimizer = th.optim.Adam(model.parameters(), lr=lr)
    scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs("models", exist_ok=True)
    ckpt_path = os.path.join("models", name + ".pt")
    best_val = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        start, running_loss, seen = time.time(), 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        scheduler.step()

        val1, val5 = evaluate(model, val_loader, device)
        print(f"epoch {epoch:2d}/{epochs} | loss {running_loss / seen:.3f} | "
              f"val top1 {val1:.3f} top5 {val5:.3f} | {time.time() - start:.0f}s")
        if val1 > best_val:
            best_val = val1
            th.save(model.state_dict(), ckpt_path)

    model.load_state_dict(th.load(ckpt_path, map_location=device))
    test1, test5 = evaluate(model, test_loader, device)
    print(f"Best val top1 {best_val:.3f} | TEST top1 {test1:.3f} top5 {test5:.3f} | saved {ckpt_path}")

    new_file = not os.path.exists("results.csv")
    with open("results.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["run", "pretrained", "augment", "seed", "epochs", "val_top1", "test_top1", "test_top5"])
        w.writerow([name, pretrained, not args.no_augment, args.seed, epochs,
                    f"{best_val:.4f}", f"{test1:.4f}", f"{test5:.4f}"])


if __name__ == "__main__":  
    main()
