
from bs4 import BeautifulSoup
import urllib.request
import urllib.parse
from datetime import datetime
from typing import List
from url_features import URLFeatureExtractor

class DomainFeatures(URLFeatureExtractor):
    """Extract domain-based features using WHOIS and traffic data"""
    
    def __init__(self, url: str, whois_module=None):
        super().__init__(url)
        self.whois_module = whois_module
        self.domain_info = None
        self.dns_available = self._check_dns()
    
    def _check_dns(self) -> bool:
        """Check if DNS record is available"""
        if self.whois_module is None:
            return False
        try:
            self.domain_info = self.whois_module.whois(self.domain)
            return True
        except Exception:
            return False
    
    def get_dns_record(self) -> int:
        """Check DNS record availability"""
        return 0 if self.dns_available else 1
    
    def get_web_traffic(self) -> int:
        """Check Alexa rank for web traffic analysis"""
        try:
            encoded_url = urllib.parse.quote(self.url)
            alexa_url = f"http://data.alexa.com/data?cli=10&dat=s&url={encoded_url}"
            response = urllib.request.urlopen(alexa_url).read()
            rank_data = BeautifulSoup(response, "xml").find("REACH")
            
            if rank_data is None:
                return 1
            
            # Safely extract rank attribute
            rank_attr = rank_data.get('RANK')
            if rank_attr is None:
                return 1
            
            # Convert to int (handle both string and list cases)
            rank_str = rank_attr[0] if isinstance(rank_attr, list) else rank_attr
            rank = int(str(rank_str))
            
            return 1 if rank < 100000 else 0
        except Exception:
            return 1
    
    def get_domain_age(self) -> int:
        """Calculate domain age from creation to expiration"""
        if not self.dns_available or self.domain_info is None:
            return 1
        
        try:
            creation_date = self.domain_info.creation_date
            expiration_date = self.domain_info.expiration_date
            
            # Handle string dates
            if isinstance(creation_date, str):
                creation_date = datetime.strptime(creation_date, '%Y-%m-%d')
            if isinstance(expiration_date, str):
                expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")
            
            # Handle None or list types
            if creation_date is None or expiration_date is None:
                return 1
            if isinstance(creation_date, list) or isinstance(expiration_date, list):
                return 1
            
            age_days = abs((expiration_date - creation_date).days)
            age_months = age_days / 30
            
            return 1 if age_months < 6 else 0
        except Exception:
            return 1
    
    def get_domain_end(self) -> int:
        """Calculate remaining time until domain expiration"""
        if not self.dns_available or self.domain_info is None:
            return 1
        
        try:
            expiration_date = self.domain_info.expiration_date
            
            # Handle string dates
            if isinstance(expiration_date, str):
                expiration_date = datetime.strptime(expiration_date, "%Y-%m-%d")
            
            # Handle None or list types
            if expiration_date is None or isinstance(expiration_date, list):
                return 1
            
            today = datetime.now()
            remaining_days = abs((expiration_date - today).days)
            remaining_months = remaining_days / 30
            
            return 0 if remaining_months < 6 else 1
        except Exception:
            return 1
    
    def extract_all(self) -> List[int]:
        """Extract all domain-based features"""
        return [
            self.get_dns_record(),
            self.get_web_traffic(),
            self.get_domain_age(),
            self.get_domain_end()
        ]
