#!/usr/bin/env python3
"""
Orchestrator for the Phishing Detection System.

Usage:
    python run.py --config-path=settings.yaml --all
"""

import argparse
from logging import Logger
import subprocess
import sys
import uvicorn
from scripts import evaluate, run_scraper
from src.config.loader import load_config, validate_config
from src.config.loader import Config
from src.config.logger import get_logger
from src.config.types import TrainingMode
from src.training import phishing_train


def run_scraper_step(config: Config, logger: Logger) -> None:
    """Run the data scraper step."""
    logger.info("Running scraper...")
    run_scraper.scraper(config=config)


def train_model_step(config: Config, logger: Logger) -> None:
    """Train the phishing detection model."""
    logger.info(f"Training model '{config.training.model_name}'...")
    phishing_train.train(config=config)


def evaluate_model_step(config: Config, logger: Logger) -> None:
    """Evaluate the trained model."""
    logger.info("Evaluating model...")
    evaluate.evaluate(config=config)


def run_api(config: Config, logger: Logger) -> None:
    """Run the FastAPI backend server."""
    logger.info(f"Starting FastAPI server on {config.api.host}:{config.api.port} (reload={config.api.reload})...")
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
    )


def run_web(config: Config, logger: Logger) -> None:
    """Run the Streamlit web UI."""
    logger.info(f"Starting Streamlit UI on {config.web.host}:{config.web.port}...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "src/web/app.py",
            "--server.address",
            config.web.host,
            "--server.port",
            str(config.web.port),
        ],
        check=True,
    )


def main(argv: list[str] | None = None) -> None:
    """Command-line entrypoint for the orchestrator."""
    parser = argparse.ArgumentParser(description="Phishing Detection Orchestrator")
    parser.add_argument(
        "--config-path", 
        type=str,
        default="src/config/settings.yaml",
        help="Path to YAML config file",
        metavar=""
    )
    parser.add_argument("--scraper", action="store_true", help="Run scraper")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in TrainingMode],
        help="Select training mode (file or folder)"
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--api", action="store_true", help="Run FastAPI backend")
    parser.add_argument("--web", action="store_true", help="Run Streamlit UI")
    parser.add_argument("--all", action="store_true", help="Run all steps sequentially")

    args = parser.parse_args(argv)

    config = load_config(args.config_path)
    validate_config(config)
    logger = get_logger(config, enable=config.app.enable_logging)

    if args.all:
        logger.info("User selected --all: running scraper, train, and evaluate")
        run_scraper_step(config, logger=logger)
        train_model_step(config, logger=logger)
        evaluate_model_step(config, logger=logger)
    else:
        if args.scraper:
            logger.info("User selected --scraper: running scraper")
            run_scraper_step(config, logger=logger)
        if args.train:
            logger.info("User selected --train: training model")
            train_model_step(config, logger=logger)
        if args.evaluate:
            logger.info("User selected --evaluate: evaluating model")
            evaluate_model_step(config, logger=logger)
        if args.api:
            logger.info("User selected --api: starting FastAPI server")
            run_api(config, logger=logger)
        if args.web:
            logger.info("User selected --web: starting Streamlit UI")
            run_web(config, logger=logger)


if __name__ == "__main__":
    main()
