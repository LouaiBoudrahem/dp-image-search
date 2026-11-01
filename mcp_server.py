import csv
import difflib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from mcp.server.fastmcp import FastMCP

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
MODEL_PATH = BASE / "models" / "resnet18_pretrained_seed0.pt"

with open(OUT / "index.csv", encoding="utf-8") as f:
    ROWS = list(csv.DictReader(f))
for r in ROWS:
    r["path"] = r["path"].replace("\\", "/")  
    r["id"] = int(r["id"])
    r["confidence"] = float(r["confidence"])
    r["file"] = Path(r["path"]).stem
CLASSES = json.load(open(OUT / "classes.json", encoding="utf-8"))
EMB = np.load(OUT / "embeddings.npy")
EMB = EMB / (np.linalg.norm(EMB, axis=1, keepdims=True) + 1e-8)
BY_FILE = {r["file"].lower(): r for r in ROWS}

_model = None

mcp = FastMCP("pet-image-search")


def find_class(label):
    norm = lambda s: s.lower().replace("_", " ").strip()
    names = {norm(c): c for c in CLASSES}
    q = norm(label)
    if q in names:
        return names[q]
    partial = [c for n, c in names.items() if q in n]
    if len(partial) == 1:
        return partial[0]
    close = difflib.get_close_matches(q, names.keys(), n=1, cutoff=0.6)
    return names[close[0]] if close else None


def unknown_class(label):
    return {"error": f"Unknown breed '{label}'. Known breeds: {', '.join(CLASSES)}"}


def short(r):
    return {"id": r["id"], "file": r["file"], "predicted": r["predicted_label"],
            "confidence": round(r["confidence"], 3), "true_label": r["true_label"]}


def run_model(path):
    global _model
    import torch as th
    from PIL import Image
    from torchvision import transforms
    sys.path.insert(0, str(BASE))
    from train import MEAN, STD, build_model, get_device

    device = get_device()
    if _model is None:
        _model = build_model(pretrained=False)
        _model.load_state_dict(th.load(MODEL_PATH, map_location=device))
        _model.to(device).eval()
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with th.no_grad():
        fc = _model.fc
        _model.fc = th.nn.Identity()
        e = _model(x)
        _model.fc = fc
        p = fc(e).softmax(dim=1)[0].cpu().numpy()
    e = e[0].cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-8), p


def resolve_image(image):
    s = str(image).strip()
    if s.isdigit() and int(s) < len(ROWS):
        r = ROWS[int(s)]
        return r, EMB[r["id"]], None
    key = Path(s).stem.lower()
    if key in BY_FILE:
        r = BY_FILE[key]
        return r, EMB[r["id"]], None
    p = Path(s) if Path(s).is_absolute() else BASE / s
    if p.is_file():
        e, probs = run_model(p)
        return None, e, probs
    raise ValueError(f"Image '{image}' not found: give an id (0-{len(ROWS) - 1}), a file name like "
                     f"'Abyssinian_1', or a path to an image file.")


@mcp.tool()
def classify(image: str) -> dict:
    """Predict the breed of one image. Returns the 3 most likely breeds with their probability.
    image: an image id (e.g. "42"), a file name (e.g. "Abyssinian_1") or a path to an image file."""
    try:
        r, e, probs = resolve_image(image)
    except ValueError as err:
        return {"error": str(err)}
    if probs is None:
        if not MODEL_PATH.exists():
            return {"image": short(r)}
        _, probs = run_model(BASE / r["path"])
    top = np.argsort(-probs)[:3]
    result = {"top3": [{"breed": CLASSES[i], "probability": round(float(probs[i]), 3)} for i in top]}
    if r is not None:
        result["image"] = short(r)
    return result


@mcp.tool()
def find_similar(image: str, k: int = 5) -> dict:
    """Find the k images that look most like the given image (nearest neighbours of the embeddings).
    image: an image id, a file name (e.g. "Abyssinian_1") or a path to an image file. k: how many (default 5)."""
    try:
        r, e, _ = resolve_image(image)
    except ValueError as err:
        return {"error": str(err)}
    k = max(1, min(int(k), 50))
    sim = EMB @ e
    if r is not None:
        sim[r["id"]] = -np.inf
    best = np.argsort(-sim)[:k]
    return {"query": short(r) if r else str(image),
            "similar": [{**short(ROWS[i]), "similarity": round(float(sim[i]), 3)} for i in best]}


@mcp.tool()
def filter_images(label: str, min_confidence: float = 0.0, max_results: int = 10) -> dict:
    """List images the model predicted as a given breed, with at least a minimum confidence.
    label: breed name (e.g. "Persian"). min_confidence: 0 to 1, e.g. 0.8 for 80% (default 0).
    max_results: how many examples to list (default 10). Also returns the total count."""
    c = find_class(label)
    if c is None:
        return unknown_class(label)
    m = float(min_confidence)
    if m > 1:
        m /= 100
    hits = [r for r in ROWS if r["predicted_label"] == c and r["confidence"] >= m]
    hits.sort(key=lambda r: -r["confidence"])
    return {"breed": c, "min_confidence": m, "count": len(hits),
            "examples": [short(r) for r in hits[:max(1, int(max_results))]]}


@mcp.tool()
def count_by(label: str = "") -> dict:
    """Count how many images the model predicted for each breed.
    label: a breed name to get only that breed's count, or empty for all 37 breeds."""
    counts = Counter(r["predicted_label"] for r in ROWS)
    if label:
        c = find_class(label)
        if c is None:
            return unknown_class(label)
        return {"breed": c, "predicted_count": counts[c],
                "true_count": sum(r["true_label"] == c for r in ROWS), "total_images": len(ROWS)}
    return {"total_images": len(ROWS), "predicted_counts": dict(counts.most_common())}


@mcp.tool()
def most_confused(label: str) -> dict:
    """Show which breeds the model confuses most with a given breed, on the test images.
    label: breed name. Returns the breeds it gets mistaken for, in both directions, with example images."""
    c = find_class(label)
    if c is None:
        return unknown_class(label)
    test = [r for r in ROWS if r["split"] == "test"]
    as_other = [r for r in test if r["true_label"] == c and r["predicted_label"] != c]
    others_as = [r for r in test if r["predicted_label"] == c and r["true_label"] != c]
    return {
        "breed": c,
        "test_images_of_breed": sum(r["true_label"] == c for r in test),
        "mistaken_for": dict(Counter(r["predicted_label"] for r in as_other).most_common(5)),
        "confused_with_it": dict(Counter(r["true_label"] for r in others_as).most_common(5)),
        "example_errors": [short(r) for r in (as_other + others_as)[:5]],
    }


def self_test():
    first = ROWS[0]["file"]
    print("classes:", len(CLASSES), "| images:", len(ROWS))
    for name, call in [
        ("count_by('pug')", lambda: count_by("pug")),
        ("filter_images('persian', 0.9, 3)", lambda: filter_images("persian", 0.9, 3)),
        (f"find_similar('{first}', 3)", lambda: find_similar(first, 3)),
        ("most_confused('staffordshire')", lambda: most_confused("staffordshire")),
        (f"classify('{first}')", lambda: classify(first)),
    ]:
        print(f"\n>>> {name}\n{json.dumps(call(), indent=1)}")


if __name__ == "__main__":
    os.chdir(BASE)
    if "--test" in sys.argv:
        self_test()
    else:
        mcp.run()
