import argparse
import glob
import json
import os
import random
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from train import MEAN, STD, build_model, get_device

TEST_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


def load_model(path, device):
    model = build_model(pretrained=False)
    model.load_state_dict(th.load(path, map_location=device))
    return model.to(device).eval()


@th.no_grad()
def predict(model, loader, device):
    """Returns embeddings (N, 512), probabilities (N, 37), labels (N,)."""
    fc = model.fc
    model.fc = nn.Identity()  
    embs, probs, labels = [], [], []
    for images, y in loader:
        e = model(images.to(device))
        embs.append(e.cpu())
        probs.append(fc(e).softmax(dim=1).cpu())
        labels.append(y)
    model.fc = fc
    return th.cat(embs).numpy(), th.cat(probs).numpy(), th.cat(labels).numpy()


def precision_at_k(features, labels, k=5):
    """Share of the k nearest neighbours (cosine similarity, excluding itself) with the same class."""
    f = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    sim = f @ f.T
    np.fill_diagonal(sim, -np.inf)
    nn_idx = np.argsort(-sim, axis=1)[:, :k]
    return (labels[nn_idx] == labels[:, None]).mean()


def topk_acc(probs, labels, k):
    top = np.argsort(-probs, axis=1)[:, :k]
    return (top == labels[:, None]).any(axis=1).mean()


def mean_std(values):
    m = statistics.mean(values) * 100
    s = statistics.stdev(values) * 100 if len(values) > 1 else 0.0
    return f"{m:.1f} ± {s:.1f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--best", default="models/resnet18_pretrained_seed0.pt",
                   help="model used for detailed analysis and export")
    args = p.parse_args()

    device = get_device()
    os.makedirs("outputs", exist_ok=True)
    th.manual_seed(0)
    random.seed(0)

    trainval = datasets.OxfordIIITPet(args.data_dir, split="trainval", download=True, transform=TEST_TF)
    test = datasets.OxfordIIITPet(args.data_dir, split="test", download=True, transform=TEST_TF)
    classes = test.classes
    kw = dict(batch_size=64, num_workers=min(4, os.cpu_count() or 1))
    test_loader = DataLoader(test, shuffle=False, **kw)

    lines = ["Test set results (mean ± std over seeds)", ""]
    groups = {}
    for path in sorted(glob.glob("models/*.pt")):
        run = os.path.basename(path)[:-3]
        group = run.rsplit("_seed", 1)[0]
        emb, probs, y = predict(load_model(path, device), test_loader, device)
        r = (topk_acc(probs, y, 1), topk_acc(probs, y, 5), precision_at_k(emb, y, 5))
        groups.setdefault(group, []).append(r)
        print(f"{run}: top1 {r[0]:.3f} top5 {r[1]:.3f} precision@5 {r[2]:.3f}")
    for group, rs in groups.items():
        lines.append(f"{group} ({len(rs)} seeds): top1 {mean_std([r[0] for r in rs])}% | "
                     f"top5 {mean_std([r[1] for r in rs])}% | precision@5 {mean_std([r[2] for r in rs])}%")

    # ---- 2. Baselines ----
    train_labels = np.array(trainval._labels)
    test_labels = np.array(test._labels)
    majority = np.bincount(train_labels).argmax()
    lines.append(f"baseline majority class ('{classes[majority]}'): top1 {(test_labels == majority).mean() * 100:.1f}%")

    pixel_tf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    pixel_test = datasets.OxfordIIITPet(args.data_dir, split="test", transform=pixel_tf)
    pixels = th.cat([x.flatten(1) for x, _ in DataLoader(pixel_test, **kw)]).numpy()
    lines.append(f"baseline raw-pixel search (32x32): precision@5 {precision_at_k(pixels, test_labels, 5) * 100:.1f}%")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open("outputs/summary.txt", "w", encoding="utf-8") as f:
        f.write(summary + "\n")

    model = load_model(args.best, device)
    run = os.path.basename(args.best)[:-3]
    emb, probs, y = predict(model, test_loader, device)
    pred = probs.argmax(axis=1)

    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    pairs = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes))
             for j in range(len(classes)) if i != j and cm[i, j] > 0]
    pairs.sort(reverse=True)
    with open(f"outputs/report_{run}.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y, pred, target_names=classes, digits=3))
        f.write("\nMost confused (true -> predicted: count)\n")
        for n, a, b in pairs[:10]:
            f.write(f"  {a} -> {b}: {n}\n")
    print("\nMost confused:", *[f"{a} -> {b} ({n})" for n, a, b in pairs[:5]], sep="\n  ")

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix – {run}")
    fig.tight_layout()
    fig.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close(fig)

    wrong = np.where(pred != y)[0]
    sample = random.sample(list(wrong), min(12, len(wrong)))
    fig, axes = plt.subplots(3, 4, figsize=(14, 11))
    for ax, i in zip(axes.flat, sample):
        ax.imshow(plt.imread(test._images[i]))
        ax.set_title(f"true: {classes[y[i]]}\npred: {classes[pred[i]]} ({probs[i, pred[i]]:.0%})", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig("outputs/errors.png", dpi=120)
    plt.close(fig)

    all_data = ConcatDataset([trainval, test])
    emb, probs, y = predict(model, DataLoader(all_data, shuffle=False, **kw), device)
    paths = [p.as_posix() for p in trainval._images] + [p.as_posix() for p in test._images]
    splits = ["trainval"] * len(trainval) + ["test"] * len(test)
    np.save("outputs/embeddings.npy", emb.astype(np.float32))
    with open("outputs/index.csv", "w", encoding="utf-8") as f:
        f.write("id,path,split,true_label,predicted_label,confidence\n")
        for i in range(len(paths)):
            k = probs[i].argmax()
            f.write(f"{i},{paths[i]},{splits[i]},{classes[y[i]]},{classes[k]},{probs[i, k]:.4f}\n")
    with open("outputs/classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, indent=1)
    print(f"\nExported {len(paths)} images to outputs/ (index.csv, embeddings.npy, classes.json)")


if __name__ == "__main__":
    main()
