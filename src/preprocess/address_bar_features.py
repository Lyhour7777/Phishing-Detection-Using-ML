from url_features import URLFeatureExtractor
import re
import ipaddress
from typing import List

class AddressBarFeatures(URLFeatureExtractor):
    """Extract features from URL address bar characteristics"""
    
    SHORTENING_SERVICES = (
        r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
        r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
        r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
        r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|"
        r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|"
        r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|"
        r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|"
        r"tr\.im|link\.zip\.net"
    )
    
    def has_ip_address(self) -> int:
        """Check if URL contains IP address instead of domain name"""
        try:
            ipaddress.ip_address(self.parsed_url.netloc)
            return 1  # Phishing
        except ValueError:
            return 0  # Legitimate
    
    def has_at_symbol(self) -> int:
        """Check for '@' symbol in URL"""
        return 1 if "@" in self.url else 0
    
    def get_url_length(self) -> int:
        """Check if URL length is suspicious (>= 54 chars)"""
        return 1 if len(self.url) >= 54 else 0
    
    def get_url_depth(self) -> int:
        """Calculate number of sub-pages based on '/' count"""
        path_parts = self.parsed_url.path.split('/')
        return sum(1 for part in path_parts if len(part) > 0)
    
    def has_redirection(self) -> int:
        """Check for '//' redirection in URL path"""
        pos = self.url.rfind('//')
        if pos > 6:
            return 1 if pos > 7 else 0
        return 0
    
    def has_http_in_domain(self) -> int:
        """Check if 'http' or 'https' appears in domain"""
        return 1 if 'https' in self.domain else 0
    
    def uses_url_shortening(self) -> int:
        """Check if URL uses shortening services"""
        match = re.search(self.SHORTENING_SERVICES, self.url)
        return 1 if match else 0
    
    def has_prefix_suffix(self) -> int:
        """Check for '-' in domain (prefix/suffix separator)"""
        return 1 if '-' in self.domain else 0
    
    def extract_all(self) -> List[int]:
        """Extract all address bar features"""
        return [
            self.has_ip_address(),
            self.has_at_symbol(),
            self.get_url_length(),
            self.get_url_depth(),
            self.has_redirection(),
            self.has_http_in_domain(),
            self.uses_url_shortening(),
            self.has_prefix_suffix()
        ]