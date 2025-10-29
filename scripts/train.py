"""
Hugging Face training script for phishing detection.
"""
from pathlib import Path
import pandas as pd
from transformers import Trainer, TrainingArguments
from src.config.loader import Config
from src.config.logger import get_logger
from src.models.model import HuggingFaceModel
from src.config.types import TrainingMode

def train(config: Config, mode: TrainingMode | None = None):
    """Train phishing detection model using CSV or folder data."""
    logger = get_logger()
    # Override mode if provided
    if isinstance(config.training.mode, str):
        config.training.mode = TrainingMode(config.training.mode)
    if mode:
        config.training.mode = mode

    logger.info(f"Training mode: {config.training.mode.value}")

    # Load data
    if config.training.mode == TrainingMode.FILE:
        data_path = Path(config.training.data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"{data_path} not found. Provide a valid CSV file.")

        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} rows for training from file.")

        # Map string labels to integers using config
        df["label_id"] = df["label"].map(config.model.labels)

    elif config.training.mode == TrainingMode.FOLDER:
        # Folder-based training can be implemented if using images
        raise NotImplementedError("Folder-based training not implemented yet.")

    else:
        raise ValueError(f"Unsupported training mode: {config.training.mode}")

    # Initialize model
    model = HuggingFaceModel(config)
    model.load()

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

    import torch
    class PhishingDataset(torch.utils.data.Dataset):
        """Phishing Dataset"""
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {key: val[idx] for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    dataset = PhishingDataset(encodings, labels)
    save_path = Path(config.training.save_model_dir) / config.training.model_name
    save_path.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(save_path),
        num_train_epochs=config.training.epochs,
        per_device_train_batch_size=config.training.batch_size,
        learning_rate=float(config.training.learning_rate),
        logging_steps=config.training.logging_steps,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        # eval_strategy="steps",
    )

    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=dataset
    )

    logger.info("Starting fine-tuning...")
    trainer.train()
    trainer.save_model(save_path)
    logger.info(f"Saved fine-tuned model to {config.training.save_model_dir}")


if __name__ == "__main__":
    from src.config.loader import load_config
    cfg = load_config("src/config/settings.yaml")
    train(cfg)
