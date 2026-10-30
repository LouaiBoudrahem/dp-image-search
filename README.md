# Image search assistant (DP project)

Chatbot (LLM via Ollama + MCP server) that explores the Oxford-IIIT Pets images using a fine-tuned ResNet-18.

## Setup
```
pip install torch torchvision scikit-learn matplotlib numpy
```
(For an NVIDIA GPU, install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/.)
or
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu12
```

## Image model
- `python train.py` trains a pretrained ResNet-18 (seed 0). Options: `--seed N`, `--scratch`, `--epochs N`.
- `run_all.bat` runs all 6 runs (3 seeds pretrained + 3 seeds scratch).
- Results go to `results.csv`, weights to `models/` (not in git: too big).

To train the model:
```
python train.py
```

Generate database files for the chatbot:        
```
python evaluate.py
```

## Output for the MCP tools (coming next)
- `index.csv`: `path, true_label, predicted_label, confidence`, one row per image
- `embeddings.npy`: one 512-number row per image, same order as `index.csv`
- `classes.json`: the 37 class names
