"""
Sample scraper script for phishing data.
This is a placeholder; replace with actual scraping logic.
"""
from pathlib import Path
import time
import pandas as pd
from src.config.loader import Config, load_config

def scraper(config: Config):
    """Scrape phishing data."""
    print("[INFO] Running scraper...", config)
    data_dir = Path(config.paths.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    data = [
        {"url": "http://example.com/phishing", "label": "phishing"},
        {"url": "http://example.com/safe", "label": "safe"}
    ]
    df = pd.DataFrame(data)
    df.to_csv(data_dir / "sample_data.csv", index=False)
    time.sleep(1)
    print(f"[INFO] Scraped {len(data)} entries and saved to {data_dir}")

if __name__ == "__main__":
    cfg = load_config("src/config/settings.yaml")
    scraper(cfg)
