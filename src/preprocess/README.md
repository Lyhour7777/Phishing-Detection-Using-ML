# Usage Example
```bash
extractor = PhishingFeatureExtractor()

# Show extractor info
print(extractor)

# Extract features for a single URL
url = "http://91.239.25.28:6892"
features = extractor.extract(url)
print(features)

# Show explained feature dictionary
info = extractor.explain(url)
for k, v in info.items():
    print(f"{k:<30}: {v}")

# Extract features for a batch of URLs
urls = ["https://www.google.com", "http://91.239.25.28:6892"]
batch_features = extractor.batch_extract(urls)
```
