import os
import pandas as pd
from src.config.loader import Config
from src.preprocess.decomposeURL import PhishingFeatureExtractor

def extractor(config: Config, phishing_n: int = None, legit_n: int = None):
    data_dir = os.path.abspath(config.paths.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    data_path = os.path.join(data_dir, config.paths.input_csv)
    data = pd.read_csv(data_path)

    if phishing_n is not None and legit_n is not None:
        phishing_sample = data[data['label'] == 0].sample(n=phishing_n, random_state=42)
        legit_sample = data[data['label'] == 1].sample(n=legit_n, random_state=42)
        dataset = pd.concat([phishing_sample, legit_sample]).reset_index(drop=True)
    else:
        dataset = data.copy().reset_index(drop=True)

    extractor_obj = PhishingFeatureExtractor()

    feature_rows = []
    failed = []

    for url in dataset["url"]:
        try:
            feature_rows.append(extractor_obj.extract(url))
        except Exception as e:
            feature_rows.append([None] * extractor_obj.total_features)
            failed.append({"url": url, "error": str(e)})

    feature_df = pd.DataFrame(
        feature_rows,
        columns=extractor_obj.FEATURE_NAMES
    )

    dataset = pd.concat([dataset, feature_df], axis=1)

    output_path = os.path.join(data_dir, config.paths.output_csv)
    dataset.to_csv(output_path, index=False)

    if failed:
        error_path = os.path.join(data_dir, "extractor_errors.csv")
        pd.DataFrame(failed).to_csv(error_path, index=False)
        print(f"[WARN] {len(failed)} URLs failed. Logged to {error_path}")

    print(f"Dataset saved to {output_path} ({len(dataset)} URLs processed)")
    return dataset