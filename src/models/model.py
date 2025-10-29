"""Base and Hugging Face models for phishing detection."""
from pathlib import Path
from typing import List, Optional
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from src.config.logger import get_logger, Config
from src.config.types import ModelProvider


class BaseModel:
    """
    Abstract base class for all models.
    Attributes:
        logger (logging.Logger): Logger instance.
        config (Config): Configuration object.
    """
    def __init__(self, config: Config) -> None:
        """Initialize base model with config and logger."""
        self.logger = get_logger()
        self.config = config

    def load(self, fine_tuned_path: str | None = None) -> None:
        """Load model and tokenizer. Must be implemented by subclasses."""
        raise NotImplementedError

    def train(self, dataset: Optional[object] = None) -> None:
        """Train or fine-tune the model. Must be implemented by subclasses."""
        raise NotImplementedError

    def predict(self, inputs: List[str]) -> List[int]:
        """Predict on new inputs. Must be implemented by subclasses."""
        raise NotImplementedError


class HuggingFaceModel(BaseModel):
    """Hugging Face transformer-based model for sequence classification."""

    def __init__(self, config: Config) -> None:
        """Initialize Hugging Face model with tokenizer and model placeholders."""
        super().__init__(config)
        self.tokenizer = None
        self.model = None

    def load(self, fine_tuned_path: str | None = None) -> None:
        """Load pretrained tokenizer and model, optionally from fine-tuned path."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        if fine_tuned_path and Path(fine_tuned_path).exists():
            self.model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_path)
            self.logger.info(f"Loaded fine-tuned model from: {fine_tuned_path}")
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model.name,
                num_labels=self.config.model.num_classes
            )
            self.logger.info(f"Loaded base model: {self.config.model.name}")

    def train(self, dataset: Optional[object] = None, resume_checkpoint: str | None = None) -> None:
        """Fine-tune the model on a dataset (HuggingFace Dataset or compatible)."""
        save_path = Path(self.config.training.save_model_dir) / self.config.training.model_name
        save_path.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(save_path),
            num_train_epochs=self.config.training.epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            learning_rate=float(self.config.training.learning_rate),
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            evaluation_strategy="steps" if self.config.training.load_best_model_at_end else "no",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )

        self.logger.info("Starting fine-tuning...")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
        trainer.save_model(save_path)
        self.logger.info(f"Saved fine-tuned model to {save_path}")

    def predict(self, inputs: List[str]) -> List[int]:
        """Tokenize input texts and return predicted labels."""
        tokenized = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.config.training.max_seq_length,
            return_tensors="pt"
        )
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**tokenized)
        return outputs.logits.argmax(dim=-1).tolist()

def get_model(config: Config) -> BaseModel:
    """Return model instance based on provider."""
    provider = ModelProvider(config.model.provider.lower())
    if provider == ModelProvider.HUGGINGFACE:
        return HuggingFaceModel(config)
    elif provider == ModelProvider.LOCAL:
        raise NotImplementedError("Local model not implemented yet")
    elif provider == ModelProvider.OPENAI:
        raise NotImplementedError("OpenAI model not implemented yet")
    else:
        raise ValueError(f"Unknown provider: {config.model.provider}")
