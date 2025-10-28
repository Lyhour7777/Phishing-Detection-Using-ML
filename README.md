# Phishing Detection System

## Overview

This project is a **comprehensive phishing detection system** that detects phishing URLs and emails using a **hybrid machine learning approach**. It combines **URL features, email text analysis, and metadata** with deep learning models for high accuracy.

The system includes:

* **Training and evaluation pipelines** for model development
* **FastAPI endpoints** for real-time predictions
* **Streamlit dashboard** for interactive exploration
* **MLOps-ready structure** for deployment on Google Cloud (Vertex AI, Cloud Run)

## Features

* **Hybrid Model Architecture**:

  * URL Encoder (CNN/LSTM)
  * Text Encoder (BERT / DistilBERT)
  * Metadata Encoder (Dense Network)
  * Feature Fusion + Classifier

* **Web Interface**:

  * Streamlit dashboard to upload emails or URLs and view predictions

* **API Interface**:

  * FastAPI server for programmatic access
  * `/predict` endpoint for phishing probability
  * `/health` endpoint for monitoring

* **MLOps-Ready**:

  * Dockerized for reproducible deployments
  * Cloud Build / CI/CD integration
  * Logging, monitoring, and experiment tracking

## Project Structure

```
phishing_detection/
│
├── data/                   # Datasets
├── notebooks/              # Exploration & experiments
├── src/                    # Source code
│   ├── config/             # Configuration files
│   ├── data/               # Preprocessing & feature extraction
│   ├── models/             # Model definitions
│   ├── training/           # Training scripts
│   ├── evaluation/         # Evaluation metrics
│   ├── utils/              # Helper functions
│   ├── web/                # Streamlit UI
│   └── api/                # FastAPI API
├── scripts/                # CLI scripts (train, evaluate, predict)
├── tests/                  # Unit tests
├── Dockerfile              # Container setup
├── pyproject.toml          # Poetry configuration
├── poetry.lock             # Locked dependencies
├── requirements.txt        # Dependencies
├── setup.py                # Python package setup
├── cloudbuild.yaml         # CI/CD for Google Cloud
└── run.py                  # Orchestrator for all components
```

## Installation

1. Clone the repository:

```bash
git clone git@github.com:Lyhour7777/Phishing-Detection-Using-ML.git
cd Phishing-Detection-Using-ML
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Optional: Set up virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

## Usage

### 1. Run Orchestrator

```bash
python run.py --train       # Train the model
python run.py --evaluate    # Evaluate the model
python run.py --api         # Launch FastAPI server
python run.py --web         # Launch Streamlit dashboard
python run.py --all         # Run all steps sequentially
```

### 2. FastAPI Endpoints

* **Predict Phishing**
  `POST /predict`
  Input: JSON `{ "url": "...", "email": "..." }`
  Output: `{ "probability": 0.85, "label": "phishing" }`

* **Health Check**
  `GET /health` → Returns `{"status": "ok"}`

### 3. Streamlit UI

* Run Streamlit:

```bash
streamlit run src/web/app.py
```

* Upload URLs or email content and view phishing predictions interactively.

## Development Guidelines

* **Linting & Code Quality**:

```bash
pylint src/ scripts/ run.py
```

* **Unit Testing**:

```bash
pytest tests/
```

* **Docker Build**:

```bash
docker build -t phishing-detection:latest .
```

* **Run with Docker Compose**:

```bash
docker-compose up
```

## Deployment (GCP / MLOps)

* **Vertex AI**: Train models and register in Vertex AI model registry
* **Cloud Run**: Deploy FastAPI and Streamlit apps
* **Cloud Build**: Automate CI/CD
* **Monitoring**: Use Cloud Monitoring / Logging for model performance and system health

## Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and commit: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

## License

This project is licensed under the **MIT License** – see `LICENSE` file for details.
