#!/usr/bin/env python3
"""
Configuration loader for the Phishing Detection System.
Uses dataclasses for structured and type-safe configuration.
"""

from dataclasses import dataclass, asdict
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
    input_csv: str
    output_csv: str
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
        # Convert provider string to ModelProvider enum if needed
        if isinstance(self.provider, str):
            self.provider = ModelProvider(self.provider.lower())
        
        # Convert mode string to TrainingMode enum if in TrainingConfig
        if hasattr(self, 'mode') and isinstance(self.mode, str):
            self.mode = TrainingMode(self.mode.lower())
        
        # Provide default labels if none are supplied
        if self.labels is None:
            self.labels = {"safe": 0, "phishing": 1}

        # Generate reverse mapping
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


def load_config(path: str | Path = "src/config/settings.yaml") -> Config:
    """
    Load YAML configuration file and parse it into dataclasses.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Config: Structured configuration object.
    """
    config_path = Path(path)
    if not config_path.exists():
        # Try to find config in parent directories
        alt_path = Path.cwd() / path
        if alt_path.exists():
            config_path = alt_path
        else:
            # Create default config file
            default_config = {
                "app": {"name": "Phishing Detection", "enable_logging": True},
                "api": {"host": "127.0.0.1", "port": 8000, "reload": True},
                "web": {"host": "127.0.0.1", "port": 8501},
                "model": {
                    "provider": "huggingface",
                    "name": "distilbert-base-uncased",
                    "temperature": 0.7,
                    "max_tokens": 512,
                    "num_classes": 2,
                    "labels": {"safe": 0, "phishing": 1}
                },
                "paths": {
                    "data_dir": "data/",
                    "input_csv": "url.csv",
                    "output_csv": "url_extractor.csv",
                    "model_dir": "models/",
                    "logs_dir": "logs/",
                    "output_dir": "outputs/"
                },
                "training": {
                    "model_name": "phishing-detector",
                    "pretrained_model": "distilbert-base-uncased",
                    "data_file": None,
                    "data_dir": None,
                    "mode": "file",
                    "batch_size": 16,
                    "epochs": 10,
                    "learning_rate": 5e-5,
                    "max_seq_length": 512,
                    "weight_decay": 0.01,
                    "warmup_steps": 100,
                    "save_model_dir": "models/",
                    "logging_steps": 50,
                    "save_steps": 500,
                    "evaluation_strategy": "steps",
                    "save_total_limit": 2,
                    "load_best_model_at_end": False,
                    "image_size": [128, 128]
                }
            }
            
            # Create directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, "w", encoding="utf-8") as file:
                yaml.dump(default_config, file, default_flow_style=False)
            
            print(f"📝 Created default config file at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    # Handle string to enum conversions
    if "model" in raw and "provider" in raw["model"]:
        raw["model"]["provider"] = ModelProvider(raw["model"]["provider"].lower())
    
    if "training" in raw and "mode" in raw["training"]:
        raw["training"]["mode"] = TrainingMode(raw["training"]["mode"].lower())

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
    warnings = []
    
    # Ensure critical config values exist
    if not config.app.name:
        warnings.append("Missing application name in 'app' section.")

    # Check API configuration
    if config.api.port != 8000:
        warnings.append(f"API port changed from default 8000 to {config.api.port}")

    if config.api.host not in ("127.0.0.1", "0.0.0.0"):
        warnings.append(f"Unusual API host: {config.api.host}")

    # Check Web UI configuration
    if config.web.port != 8501:
        warnings.append(f"Web UI port changed from default 8501 to {config.web.port}")

    if config.web.host not in ("127.0.0.1", "0.0.0.0"):
        warnings.append(f"Unusual Web UI host: {config.web.host}")

    # Check Paths
    for name, path in vars(config.paths).items():
        if not path:
            warnings.append(f"Path for '{name}' is empty in config.")

    # Check Training section
    if float(config.training.learning_rate) <= 0:
        warnings.append("Learning rate must be positive.")
    if int(config.training.epochs) <= 0:
        warnings.append("Epochs must be greater than zero.")
    if int(config.training.batch_size) <= 0:
        warnings.append("Batch size must be greater than zero.")

    # Print warnings if any
    if warnings:
        print("[CONFIG VALIDATION WARNINGS]:")
        for warning in warnings:
            print(f"  ⚠️  {warning}")
    else:
        print("✅ Configuration validated successfully.")

    return warnings


# Load configuration with error handling
try:
    CONFIG = load_config()
    CONFIG_DICT = asdict(CONFIG)  # Dictionary version for compatibility
    validate_config(CONFIG)
except Exception as e:
    print(f"❌ Error loading config: {e}")
    # Create a minimal config for fallback
    CONFIG = Config(
        app=AppConfig(name="Phishing Detection", enable_logging=True),
        api=APIConfig(host="127.0.0.1", port=8000, reload=True),
        web=WebConfig(host="127.0.0.1", port=8501),
        paths=PathsConfig(
            data_dir="data/",
            model_dir="models/",
            logs_dir="logs/",
            output_dir="outputs/"
        ),
        model=ModelConfig(
            provider=ModelProvider.HUGGINGFACE,
            name="distilbert-base-uncased",
            temperature=0.7,
            max_tokens=512,
            num_classes=2,
            labels={"safe": 0, "phishing": 1}
        ),
        training=TrainingConfig(
            model_name="phishing-detector",
            pretrained_model="distilbert-base-uncased"
        )
    )
    CONFIG_DICT = asdict(CONFIG)
    print("⚠️  Using fallback configuration")