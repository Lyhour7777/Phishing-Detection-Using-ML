import re
from urllib.parse import urlparse


class URLFeatureExtractor:
    """Base class for URL-based feature extraction"""
    
    def __init__(self, url: str):
        self.url = url
        self.parsed_url = urlparse(url)
        self.domain = self._extract_domain()
    
    def _extract_domain(self) -> str:
        """Extract domain from URL"""
        domain = self.parsed_url.netloc
        if re.match(r"^www\.", domain):
            domain = domain.replace("www.", "")
        return domain