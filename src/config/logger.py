"""log"""
import logging
from pathlib import Path
from src.config.loader import Config

_state = {"logger": None}  # module-level singleton container

def get_logger(
        config: Config = None, 
        name: str = "pfd_logger", 
        enable: bool = True
    ) -> logging.Logger:
    """Return a configured logger. Uses singleton so it’s the same everywhere."""
    if _state["logger"]:
        return _state["logger"]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.disabled = not enable

    if not logger.handlers:
        if config:
            logs_dir = Path(config.paths.logs_dir)
            logs_dir.mkdir(parents=True, exist_ok=True)

            formatter = logging.Formatter(
                fmt="[%(asctime)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # File handler
            log_file = logs_dir / f"{name}.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    _state["logger"] = logger
    return _state["logger"]
