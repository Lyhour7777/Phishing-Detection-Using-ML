from url_features import URLFeatureExtractor
import requests
from typing import List, Optional
import re

class HTMLJavaScriptFeatures(URLFeatureExtractor):
    """Extract features from HTML and JavaScript content"""
    
    def __init__(self, url: str):
        super().__init__(url)
        self.response = self._get_response()
    
    def _get_response(self) -> Optional[requests.Response]:
        """Fetch URL content"""
        try:
            return requests.get(self.url, timeout=5)
        except Exception:
            return None
    
    def has_iframe(self) -> int:
        """Check for iframe redirection"""
        if self.response is None:
            return 1
        
        if re.findall(r"[<iframe>|<frameBorder>]", self.response.text):
            return 0  # Legitimate
        return 1  # Phishing
    
    def has_mouse_over(self) -> int:
        """Check for status bar customization via onmouseover"""
        if self.response is None:
            return 1
        
        if re.findall("<script>.+onmouseover.+</script>", self.response.text):
            return 1  # Phishing
        return 0  # Legitimate
    
    def has_right_click_disabled(self) -> int:
        """Check if right-click is disabled"""
        if self.response is None:
            return 1
        
        if re.findall(r"event.button ?== ?2", self.response.text):
            return 0  # Legitimate (right-click enabled)
        return 1  # Phishing (right-click disabled)
    
    def get_forwarding_count(self) -> int:
        """Check number of URL redirections"""
        if self.response is None:
            return 1
        
        return 0 if len(self.response.history) <= 2 else 1
    
    def extract_all(self) -> List[int]:
        """Extract all HTML/JavaScript features"""
        return [
            self.has_iframe(),
            self.has_mouse_over(),
            self.has_right_click_disabled(),
            self.get_forwarding_count()
        ]