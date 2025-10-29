#!/usr/bin/env python3
"""
Configuration loader for the Phishing Detection System.
Uses dataclasses for structured and type-safe configuration.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from src.config.types import ModelProvider, TrainingMode


@dataclass
class AppConfig:
    """Application metadata."""
    name: str
    enable_logging: bool


@dataclass
class APIConfig:
    """Configuration for the FastAPI backend."""
    host: str
    port: int
    reload: bool


@dataclass
class WebConfig:
    """Configuration for the Streamlit frontend."""
    host: str
    port: int


@dataclass
class PathsConfig:
    """Directory paths for data, models, and logs."""
    data_dir: str
    model_dir: str
    logs_dir: str
    output_dir: str


@dataclass
class TrainingConfig:
    """Model training hyperparameters."""
    model_name: str
    pretrained_model: str

    # Input data
    data_file: Optional[str] = None
    data_dir: Optional[str] = None
    mode: TrainingMode = TrainingMode.FILE

    # Fine-tuning hyperparameters
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 5e-5
    max_seq_length: int = 512
    weight_decay: float = 0.01
    warmup_steps: int = 100

    # Optional image size (for vision models)
    image_size: Optional[List[int]] = None

    # Logging & saving
    save_model_dir: str = "models/"
    logging_steps: int = 50
    save_steps: int = 500
    evaluation_strategy: str = "steps"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True

@dataclass
class ModelConfig:
    """Model configuration."""
    provider: ModelProvider
    name: str
    temperature: float = 0.7
    max_tokens: int = 512
    num_classes: int = 2
    labels: Dict[str, int] = None

    def __post_init__(self):
        # Provide default labels if none are supplied
        if self.labels is None:
            self.labels = {"safe": 0, "phishing": 1}

        # Optional: generate reverse mapping
        self.id2label: Dict[int, str] = {v: k for k, v in self.labels.items()}

@dataclass
class Config:
    """Top-level configuration structure."""
    app: AppConfig
    api: APIConfig
    web: WebConfig
    paths: PathsConfig
    model: ModelConfig
    training: TrainingConfig


def load_config(path: str | Path) -> Config:
    """
    Load YAML configuration file and parse it into dataclasses.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Config: Structured configuration object.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    return Config(
        app=AppConfig(**raw["app"]),
        api=APIConfig(**raw["api"]),
        web=WebConfig(**raw["web"]),
        paths=PathsConfig(**raw["paths"]),
        model=ModelConfig(**raw["model"]),
        training=TrainingConfig(**raw["training"]),
    )

def validate_config(config: Config) -> None:
    """
    Validate the loaded configuration and warn about missing or unusual values.

    Args:
        config: The loaded configuration dataclass.
    """
    # Ensure critical config values exist
    if not config.app.name:
        print("[WARNING] Missing application name in 'app' section.")

    # Check API configuration
    if config.api.port != 8000:
        print(f"[WARNING] API port changed from default 8000 to {config.api.port}")

    if config.api.host not in ("127.0.0.1", "0.0.0.0"):
        print(f"[WARNING] Unusual API host: {config.api.host}")

    # Check Web UI configuration
    if config.web.port != 8501:
        print(f"[WARNING] Web UI port changed from default 8501 to {config.web.port}")

    if config.web.host not in ("127.0.0.1", "0.0.0.0"):
        print(f"[WARNING] Unusual Web UI host: {config.web.host}")

    # Check Paths
    for name, path in vars(config.paths).items():
        if not path:
            print(f"[WARNING] Path for '{name}' is empty in config.")

    # Check Training section
    if float(config.training.learning_rate) <= 0:
        print("[WARNING] Learning rate must be positive.")
    if int(config.training.epochs) <= 0:
        print("[WARNING] Epochs must be greater than zero.")
    if int(config.training.batch_size) <= 0:
        print("[WARNING] Batch size must be greater than zero.")
