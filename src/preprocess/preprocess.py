from extractor import BatchPhishingExtractor

if __name__ == "__main__":
    # Initialize extractor (with or without WHOIS)
    try:
        import whois
        extractor = BatchPhishingExtractor(whois_module=whois)
    except ImportError:
        print("Warning: python-whois not installed. Domain features will be limited.")
        extractor = BatchPhishingExtractor()
    
    # ============= Example 1: Single URL =============
    print("=" * 50)
    print("Example 1: Single URL")
    print("=" * 50)
    test_url = "http://example.com/test"
    features = extractor.extract_features(test_url)
    print(f"Features: {features}\n")
    
    # ============= Example 2: Multiple URLs (List) =============
    print("=" * 50)
    print("Example 2: Multiple URLs from List")
    print("=" * 50)
    urls = [
        "http://example.com",
        "https://google.com",
        "http://bit.ly/test123",
        "http://192.168.1.1/admin"
    ]
    
    # Get results as DataFrame
    df = extractor.extract_to_dataframe(urls, show_progress=True)
    print(df.head())
    print()
    
    
    # ============= Example 4: From CSV File =============
    # print("=" * 50)
    # print("Example 4: From CSV File")
    # print("=" * 50)
    # print("If you have a CSV with 'url' and 'label' columns:")
    # print("df = extractor.extract_from_csv('input.csv', url_column='url', label_column='label')")
    # print()
    
    # ============= Example 5: Save to CSV =============
    # print("=" * 50)
    # print("Example 5: Save Results to CSV")
    # print("=" * 50)
    # print("Save extracted features to CSV:")
    # print("extractor.save_to_csv(urls, 'features_output.csv')")
    # print()