"""
Sample training script.
Replace with actual ML model training.
"""
from pathlib import Path
import time
import pickle
import pandas as pd

from src.config.loader import Config
from src.config.types import TrainingMode


def train(config: Config, mode: TrainingMode | None = None):
    """Train phishing detection model using either a CSV file or a folder of images."""
    if mode:
        config.training.mode = mode

    print(f"[INFO] Training mode: {config.training.mode.value}")

    if config.training.mode == TrainingMode.FILE:
        # CSV file training
        data_path = Path(config.training.data_file)
        if not data_path.exists():
            raise FileNotFoundError(f"{data_path} not found. Provide a valid CSV file.")

        df = pd.read_csv(data_path)
        print(f"[INFO] Loaded {len(df)} rows for training from file.")

        phishing_count = (df["label"] == "phishing").sum()
        safe_count = (df["label"] == "safe").sum()

    elif config.training.mode == TrainingMode.FOLDER:
        # Folder-based training (images)
        data_dir = Path(config.training.data_dir)
        if not data_dir.exists() or not any(data_dir.iterdir()):
            raise FileNotFoundError(
                f"{data_dir} is empty or does not exist. Provide a valid folder."
            )
        # Dummy: count total image files
        image_files = list(data_dir.glob("**/*.*"))
        print(f"[INFO] Found {len(image_files)} images for training in folder.")
        phishing_count = len(image_files) // 2  # dummy split
        safe_count = len(image_files) - phishing_count

    else:
        raise ValueError(f"Unsupported training mode: {config.training.mode}")

    # Training hyperparameters
    batch_size = config.training.batch_size
    epochs = config.training.epochs
    lr = config.training.learning_rate
    print(f"[INFO] Training model '{config.training.model_name}': "
          f"batch_size={batch_size}, epochs={epochs}, lr={lr}")
    time.sleep(2)  # simulate training

    # Save dummy model info
    model_dir = Path(config.paths.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{config.training.model_name}.pkl"
    model_info = {"phishing_count": phishing_count, "safe_count": safe_count}

    with open(model_path, "wb") as f:
        pickle.dump(model_info, f)

    print(f"[INFO] Saved model to {model_path}")


if __name__ == "__main__":
    from src.config.loader import load_config
    cfg = load_config("src/config/settings.yaml")
    train(cfg)