from typing import List, Optional
from address_bar_features import AddressBarFeatures
from domain_features import DomainFeatures
from content_features import HTMLJavaScriptFeatures

class PhishingFeatureExtractor:
    """Main class to orchestrate feature extraction"""
    
    FEATURE_NAMES = [
        'Have_IP', 'Have_At', 'URL_Length', 'URL_Depth', 'Redirection',
        'https_Domain', 'TinyURL', 'Prefix/Suffix', 'DNS_Record', 
        'Web_Traffic', 'Domain_Age', 'Domain_End', 'iFrame', 
        'Mouse_Over', 'Right_Click', 'Web_Forwards'
    ]
    
    def __init__(self, whois_module=None):
        """
        Initialize the feature extractor
        
        Args:
            whois_module: Optional WHOIS module (e.g., python-whois)
        """
        self.whois_module = whois_module
    
    def extract_features(self, url: str) -> List[int]:
        """
        Extract all features from a URL
        
        Args:
            url: The URL to analyze
            
        Returns:
            List of feature values (0 or 1)
        """
        features = []
        
        # Address bar features
        address_extractor = AddressBarFeatures(url)
        features.extend(address_extractor.extract_all())
        
        # Domain features
        domain_extractor = DomainFeatures(url, self.whois_module)
        features.extend(domain_extractor.extract_all())
        
        # HTML/JavaScript features
        html_extractor = HTMLJavaScriptFeatures(url)
        features.extend(html_extractor.extract_all())
        
        return features
    
    def extract_features_dict(self, url: str) -> dict:
        """
        Extract features and return as dictionary
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dictionary mapping feature names to values
        """
        features = self.extract_features(url)
        return dict(zip(self.FEATURE_NAMES, features))
    
    
class BatchPhishingExtractor(PhishingFeatureExtractor):
    """Extended class for batch processing multiple URLs"""
    
    def extract_features_batch(self, urls: List[str], show_progress: bool = True) -> List[List[int]]:
        """
        Extract features from multiple URLs
        
        Args:
            urls: List of URLs to analyze
            show_progress: Show progress during extraction
            
        Returns:
            List of feature lists for each URL
        """
        results = []
        total = len(urls)
        
        for idx, url in enumerate(urls, 1):
            if show_progress:
                print(f"Processing {idx}/{total}: {url[:50]}...")
            
            try:
                features = self.extract_features(url)
                results.append(features)
            except Exception as e:
                if show_progress:
                    print(f"  Error: {e}")
                results.append([0] * len(self.FEATURE_NAMES))  # Default values on error
        
        return results
    
    def extract_to_dataframe(self, urls: List[str], show_progress: bool = True):
        """
        Extract features and return as pandas DataFrame
        
        Args:
            urls: List of URLs to analyze
            show_progress: Show progress during extraction
            
        Returns:
            pandas DataFrame with features
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for this method. Install with: pip install pandas")
        
        features_list = self.extract_features_batch(urls, show_progress)
        df = pd.DataFrame(features_list, columns=self.FEATURE_NAMES)
        df.insert(0, 'URL', urls)
        
        return df
    
    def extract_from_file(self, filepath: str, show_progress: bool = True):
        """
        Extract features from URLs in a text file (one URL per line)
        
        Args:
            filepath: Path to file containing URLs
            show_progress: Show progress during extraction
            
        Returns:
            pandas DataFrame with features
        """
        with open(filepath, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        if show_progress:
            print(f"Loaded {len(urls)} URLs from {filepath}")
        
        return self.extract_to_dataframe(urls, show_progress)
    
    def extract_from_csv(self, filepath: str, url_column: str = 'url', 
                        label_column: Optional[str] = None, show_progress: bool = True):
        """
        Extract features from URLs in a CSV file
        
        Args:
            filepath: Path to CSV file
            url_column: Name of column containing URLs
            label_column: Optional name of column containing labels
            show_progress: Show progress during extraction
            
        Returns:
            pandas DataFrame with features
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for this method. Install with: pip install pandas")
        
        # Read CSV
        df_input = pd.read_csv(filepath)
        urls = df_input[url_column].tolist()
        
        if show_progress:
            print(f"Loaded {len(urls)} URLs from {filepath}")
        
        # Extract features
        df_features = self.extract_to_dataframe(urls, show_progress)
        
        # Add labels if specified
        if label_column and label_column in df_input.columns:
            df_features['Label'] = df_input[label_column].values
        
        return df_features
    
    def save_to_csv(self, urls: List[str], output_filepath: str, show_progress: bool = True):
        """
        Extract features and save directly to CSV
        
        Args:
            urls: List of URLs to analyze
            output_filepath: Path to save output CSV
            show_progress: Show progress during extraction
        """
        df = self.extract_to_dataframe(urls, show_progress)
        df.to_csv(output_filepath, index=False)
        
        if show_progress:
            print(f"\nSaved results to {output_filepath}")
