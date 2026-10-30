# Image search assistant (DP project)

Chatbot (LLM via Ollama + MCP server) that explores the Oxford-IIIT Pets images using a fine-tuned ResNet-18.

## Setup
```
pip install torch torchvision scikit-learn matplotlib numpy ollama "mcp<2"
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

## Chatbot (Ollama + MCP)

Before running the chatbot, make sure you have generated the database files using evaluate.py as shown above.
* 1-Ensure the Ollama application is running on your machine.
* 2-Download the default model "tools manager" (if you haven't already):
```
ollama pull qwen2.5:7b
```
Start the interactive chatbot:
```
python chatbot.py
```
* 3- alternative Using the CRIL Ollama API
If you want to use the CRIL server instead of your local machine, use the --host flag (or the OLLAMA_HOST environment variable):
```
python chatbot.py --model llama3.1:8b --host http://HOTE_CRIL:11434
```

## Example Queries

Once the chatbot is running, try asking these questions to test the different tools:
* Test errors/hypotheses (most_confused): "Avec quelles races le modèle confond-il le plus le Staffordshire Bull Terrier, et pourquoi à ton avis ?"
* Test filtering (filter_images): "Montre-moi quelques exemples de chats Persans où le modèle est sûr de sa prédiction à plus de 90 %."
* Test counting (count_by): "Combien d'images de Carlin (pug) le modèle a-t-il trouvé au total ?"
* Test similarity (find_similar): "Trouve-moi les 3 images qui ressemblent le plus à la photo 'Abyssinian_100'."
* Test classification (classify): "Quelles sont les 3 races les plus probables selon le modèle pour le fichier

## Response Example

bot > Le modèle confond le plus le Staffordshire Bull Terrier avec l'American Pit Bull Terrier, suivi de l'American Bulldog et du Boxer. 

Hypothèse : Ces races partagent des similarités dans la morphologie et la taille, ce qui peut rendre leur distinction difficile pour le modèle. Par exemple, tous ces chiens sont de taille moyenne à grande, ont une musculature prononcée et des caractéristiques faciales similaires.