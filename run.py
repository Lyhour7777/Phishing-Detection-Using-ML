#!/usr/bin/env python3
"""
Orchestrator for the Phishing Detection System.

Usage:
    python run.py --config-path=settings.yaml --all
"""

import argparse
import subprocess
import sys
import uvicorn
from scripts import evaluate, run_scraper, train
from src.config.loader import load_config, validate_config
from src.config.loader import Config
from src.config.types import TrainingMode


def run_scraper_step(config: Config) -> None:
    """Run the data scraper step."""
    print("[INFO] Running scraper...")
    run_scraper.scraper(config=config)


def train_model_step(config: Config) -> None:
    """Train the phishing detection model."""
    print(f"[INFO] Training model '{config.training.model_name}'...")
    train.train(config=config)


def evaluate_model_step(config: Config) -> None:
    """Evaluate the trained model."""
    print("[INFO] Evaluating model...")
    evaluate.evaluate(config=config)


def run_api(config: Config) -> None:
    """Run the FastAPI backend server."""
    print(f"[INFO] Starting FastAPI server on {config.api.host}:{config.api.port} (reload={config.api.reload})...")
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
    )


def run_web(config: Config) -> None:
    """Run the Streamlit web UI."""
    print(f"[INFO] Starting Streamlit UI on {config.web.host}:{config.web.port}...")
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
        help="Path to YAML config file"
    )
    parser.add_argument("--scraper", action="store_true", help="Run scraper")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument(
        "--mode", 
        type=TrainingMode, 
        hoices=list(TrainingMode), 
        help="Select training mode (file or folder)"
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model")
    parser.add_argument("--api", action="store_true", help="Run FastAPI backend")
    parser.add_argument("--web", action="store_true", help="Run Streamlit UI")
    parser.add_argument("--all", action="store_true", help="Run all steps sequentially")

    args = parser.parse_args(argv)

    config = load_config(args.config_path)
    validate_config(config)

    if args.all:
        run_scraper_step(config)
        train_model_step(config)
        evaluate_model_step(config)
    else:
        if args.scraper:
            run_scraper_step(config)
        if args.train:
            train_model_step(config)
        if args.evaluate:
            evaluate_model_step(config)
        if args.api:
            run_api(config)
        if args.web:
            run_web(config)


if __name__ == "__main__":
    main()
