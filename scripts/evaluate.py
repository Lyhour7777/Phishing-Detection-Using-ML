from pathlib import Path
import torch
from src.models.model import HuggingFaceModel
from src.config.loader import load_config
from src.config.logger import get_logger
import pandas as pd

def evaluate(config):
    logger = get_logger()

    # Initialize and load model
    model = HuggingFaceModel(config)
    model.load()

    # Load your fine-tuned weights
    fine_tuned_path = Path(config.training.save_model_dir) / config.training.model_name
    model.model = model.model.from_pretrained(fine_tuned_path)
    logger.info(f"Loaded fine-tuned model from {fine_tuned_path}")

    # Load evaluation dataset
    df = pd.read_csv(config.training.data_file)
    logger.info(f"Loaded {len(df)} rows for evaluation.")

    # Map labels
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

    # Compute accuracy
    labels = df["label_id"].tolist()
    correct = sum(p == l for p, l in zip(preds, labels))
    accuracy = correct / len(labels)
    logger.info(f"Evaluation accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    cfg = load_config("src/config/settings.yaml")
    evaluate(cfg)