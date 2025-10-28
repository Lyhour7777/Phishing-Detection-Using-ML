"""
Sample evaluation script.
Replace with real evaluation logic.
"""
import pickle
from pathlib import Path

from src.config.loader import Config, load_config
from src.config.logger import get_logger

def evaluate(config: Config):
    """Evaluate the trained model."""
    logger = get_logger()
    model_path = Path(config.paths.model_dir) / f"{config.training.model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Train a model first.")

    logger.info("Loading model...")
    with open(model_path, "rb") as f:
        model_info = pickle.load(f)

    total = model_info["phishing_count"] + model_info["safe_count"]
    accuracy = model_info["safe_count"] / total if total > 0 else 0
    logger.info(f"Dummy evaluation accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    cfg = load_config("src/config/settings.yaml")
    evaluate(cfg)
