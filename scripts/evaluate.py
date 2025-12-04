"""Evaluate a fine-tuned Hugging Face phishing detection model."""
# Evaluate a fine-tuned model
from pathlib import Path
import torch
import pandas as pd
from src.models.model import HuggingFaceModel
from src.config.loader import load_config
from src.config.logger import get_logger

def get_latest_checkpoint(model_dir: Path) -> Path | None:
    """load checkpoint"""
    checkpoints = sorted(
        [p for p in model_dir.iterdir() if p.is_dir() and "checkpoint" in p.name],
        key=lambda x: int(x.name.split("-")[1])
    )
    return checkpoints[-1] if checkpoints else None

def evaluate(config):
    """evaluate"""
    logger = get_logger()
    save_path = Path(config.training.save_model_dir) / config.training.model_name
    checkpoint = get_latest_checkpoint(save_path)

    if checkpoint is None:
        raise ValueError("No checkpoint found to evaluate!")

    # Load fine-tuned model
    model = HuggingFaceModel(config)
    model.load(fine_tuned_path=str(checkpoint))
    logger.info("Loaded fine-tuned model from %s", checkpoint)

    # Load evaluation data
    df = pd.read_csv(config.training.data_file)
    df["label_id"] = df["label"].map(config.model.labels)
    
    # Tokenize
    encodings = model.tokenizer(
        df["text"].tolist(),
        padding=True,
        truncation=True,
        max_length=config.training.max_seq_length,
        return_tensors="pt"
    )

    # Inference
    model.model.eval()
    with torch.no_grad():
        outputs = model.model(**encodings)
        preds = outputs.logits.argmax(dim=-1).tolist()

    # Accuracy
    labels = df["label_id"].tolist()
    correct = sum(p == l for p, l in zip(preds, labels))
    accuracy = correct / len(labels)
    logger.info("Evaluation accuracy: %.2f", accuracy)

if __name__ == "__main__":
    cfg = load_config("src/config/settings.yaml")
    evaluate(cfg)