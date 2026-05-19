import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# GRIM NLP Model - Hugging Face `wiki40b_ja` Dataset Loader & Preprocessing\n",
                "\n",
                "This notebook demonstrates how to load, preprocess, and construct data loaders for the GRIM model using the Hugging Face `datasets` library with the Japanese Wikipedia dataset (`fn-aka-mur/wiki40b_ja`).\n",
                "\n",
                "### Contents:\n",
                "1. Environment Setup & Dependency Installation\n",
                "2. Direct loading of the dataset via the Hugging Face `datasets` library\n",
                "3. Creating the GRIM `TextCorpus` using our optimized HF helper\n",
                "4. Constructing validation/training data loaders with chronological (temporal) split\n",
                "5. Performing a single dummy forward pass of the GRIM model to verify convergence compatibility"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Environment Setup & Dependency Installation\n",
                "# Install the packages if not already present\n",
                "!pip install datasets tqdm torchdiffeq numpy gradio"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Loading the dataset directly via the HF datasets library\n",
                "import datasets\n",
                "\n",
                "print(\"Loading fn-aka-mur/wiki40b_ja split='train' in streaming mode...\")\n",
                "# Use streaming mode to avoid loading the entire giant dataset into memory\n",
                "dataset = datasets.load_dataset(\"fn-aka-mur/wiki40b_ja\", split=\"train\", streaming=True)\n",
                "\n",
                "# Fetch a few samples to inspect the schema\n",
                "sample_iterator = iter(dataset)\n",
                "for i in range(3):\n",
                "    sample = next(sample_iterator)\n",
                "    print(f\"\\n--- Article {i+1} ---\")\n",
                "    print(f\"Keys available: {list(sample.keys())}\")\n",
                "    # Print the first 200 characters of the text field\n",
                "    text_content = sample.get(\"text\", sample.get(\"main_text\", \"\"))\n",
                "    print(text_content[:200] + \"...\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Build GRIM TextCorpus using the project's helper\n",
                "import sys\n",
                "from pathlib import Path\n",
                "\n",
                "# Add project root to path\n",
                "ROOT = Path(\".\").resolve()\n",
                "if str(ROOT) not in sys.path:\n",
                "    sys.path.insert(0, str(ROOT))\n",
                "\n",
                "from grim.data.hf_dataset import load_hf_corpus\n",
                "\n",
                "# Load max 150,000 characters from the training split of the dataset\n",
                "print(\"Loading corpus via load_hf_corpus helper...\")\n",
                "corpus, col_name = load_hf_corpus(\n",
                "    dataset=\"fn-aka-mur/wiki40b_ja\",\n",
                "    split=\"train\",\n",
                "    max_chars=150000, # Load 150,000 characters for demo\n",
                "    streaming=True,\n",
                ")\n",
                "\n",
                "print(f\"\\nSuccess!\")\n",
                "print(f\"Source: {corpus.source}\")\n",
                "print(f\"Vocabulary size: {corpus.vocab.size}\")\n",
                "print(f\"Total characters: {len(corpus.text)}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Constructing Training and Validation Dataloaders with Chronological (Temporal) Split\n",
                "from grim.data.text import get_lm_loaders\n",
                "\n",
                "seq_len = 64\n",
                "batch_size = 16\n",
                "val_ratio = 0.3 # 30% for validation to prevent leakage on small corpus\n",
                "\n",
                "train_loader, val_loader, vocab = get_lm_loaders(\n",
                "    corpus,\n",
                "    seq_len=seq_len,\n",
                "    batch_size=batch_size,\n",
                "    val_ratio=val_ratio,\n",
                ")\n",
                "\n",
                "print(f\"DataLoaders successfully built!\")\n",
                "print(f\"Number of training batches: {len(train_loader)}\")\n",
                "print(f\"Number of validation batches: {len(val_loader)}\")\n",
                "\n",
                "# Inspect one batch\n",
                "x, y = next(iter(train_loader))\n",
                "print(f\"Input batch shape: {x.shape} (Batch size, Sequence length)\")\n",
                "print(f\"Target batch shape: {y.shape} (Batch size)\")\n",
                "print(\"\\nFirst batch sample:\")\n",
                "print(f\"Input context: '{vocab.decode(x[0].tolist())}'\")\n",
                "print(f\"Next char target: '{vocab.decode([y[0].item()])}'\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Initialize GRIM Model and Run a Dummy Forward Pass\n",
                "import torch\n",
                "from grim.config import GRIMConfig\n",
                "from grim.model import GRIM\n",
                "\n",
                "# Create config matching the fast CPU preset\n",
                "config = GRIMConfig(\n",
                "    task_mode=\"lm\",\n",
                "    V=max(vocab.size, 64),\n",
                "    device=\"cpu\",\n",
                "    seq_len=seq_len,\n",
                "    M_max=seq_len,\n",
                ")\n",
                "config.apply_fast_preset()\n",
                "\n",
                "# Instantiate the model with transition matrix T and 4-term Energy function\n",
                "print(\"Instantiating GRIM model...\")\n",
                "model = GRIM(config)\n",
                "\n",
                "# Perform a forward train step to check that it works flawlessly\n",
                "print(\"Running dummy forward pass...\")\n",
                "x_dev, y_dev = x.to(config.device), y.to(config.device)\n",
                "out = model.forward_train_lm(x_dev, y_dev)\n",
                "\n",
                "print(\"\\n--- Forward Pass Status ---\")\n",
                "print(f\"Weighted Loss: {out['loss'].item():.4f}\")\n",
                "print(f\"LM Born-rule Loss: {out['loss_obs'].item():.4f}\")\n",
                "print(f\"Target State Unitary check: {torch.abs(torch.sum(torch.abs(out['psi_T'])**2, dim=-1) - 1.0).max().item() < 1e-5}\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

output_path = Path("wiki40b_loader.ipynb")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Created {output_path.resolve()}")
