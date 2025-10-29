# scripts/train.py
from pathlib import Path
import pandas as pd
from src.config.loader import Config
from src.config.logger import get_logger
from src.models.model import HuggingFaceModel
from src.config.types import TrainingMode
from src.training.phishing_dataset import PhishingDataset

def train(config: Config, mode: TrainingMode | None = None):
    """Train phishing detection model using CSV or folder data."""
    logger = get_logger()

    # Override mode if provided
    if isinstance(config.training.mode, str):
        config.training.mode = TrainingMode(config.training.mode)
    if mode:
        config.training.mode = mode

    logger.info(f"Training mode: {config.training.mode.value}")

    # Load CSV data
    if config.training.mode == TrainingMode.FILE:
        data_path = Path(config.training.data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"{data_path} not found. Provide a valid CSV file.")

        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows for training from file.")
        df["label_id"] = df["label"].map(config.model.labels)
    else:
        raise NotImplementedError("Folder-based training not implemented yet.")

    # Initialize model
    model = HuggingFaceModel(config)

    # Check if a fine-tuned model already exists
    save_path = Path(config.training.save_model_dir) / config.training.model_name
    fine_tuned_checkpoint = save_path if save_path.exists() else None
    model.load(fine_tuned_checkpoint)

    # Tokenize dataset
    texts = df["text"].tolist()
    labels = df["label_id"].tolist()
    encodings = model.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config.training.max_seq_length,
        return_tensors="pt"
    )
    dataset = PhishingDataset(encodings, labels)

    # Fine-tune model, continue if checkpoint exists
    model.train(dataset=dataset, resume_checkpoint=fine_tuned_checkpoint)
