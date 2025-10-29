"""Base and Hugging Face models for phishing detection."""
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

    def load(self) -> None:
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

    def load(self) -> None:
        """Load pretrained tokenizer and model from Hugging Face."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model.name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model.name,
            num_labels=2  # phishing / safe
        )
        self.logger.info(f"Loaded Hugging Face model: {self.config.model.name}")

    def train(self, dataset: Optional[object] = None) -> None:
        """Fine-tune the model on a dataset (HuggingFace Dataset or compatible)."""
        training_args = TrainingArguments(
            output_dir=self.config.training.save_model_dir,
            num_train_epochs=self.config.training.epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            learning_rate=self.config.training.learning_rate,
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset
        )
        self.logger.info("Starting fine-tuning...")
        trainer.train()
        trainer.save_model(self.config.training.save_model_dir)
        self.logger.info(f"Saved fine-tuned model to {self.config.training.save_model_dir}")

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
        # implement your local model class later
        raise NotImplementedError("Local model not implemented yet")
    elif provider == ModelProvider.OPENAI:
        # implement your OpenAI wrapper later
        raise NotImplementedError("OpenAI model not implemented yet")
    else:
        raise ValueError(f"Unknown provider: {config.model.provider}")