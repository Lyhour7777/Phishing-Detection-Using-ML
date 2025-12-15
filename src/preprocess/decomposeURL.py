from urllib.parse import urlparse
import ipaddress
import re
import whois
import requests
from datetime import datetime

class PhishingFeatureExtractor:
    FEATURE_NAMES = [
        "Having IP address",
        "Having @ symbol",
        "URL length >= 54",
        "Multiple subdomains",
        "Redirection //",
        "HTTP instead of HTTPS",
        "URL shortening service",
        "Prefix/Suffix in domain",
        "DNS record missing",
        "Domain age < 6 months",
        "Domain expires in <6 months",
        "Iframe detected",
        "Mouse-over script",
        "Right-click disabled",
        "Multiple forwarding"
    ]

    SHORTENING_SERVICES = re.compile(
        r"bit\.ly|bitly\.com|tinyurl\.com|is\.gd|v\.gd|t\.co|ow\.ly|"
        r"buff\.ly|rebrand\.ly|rb\.gy|short\.io|soo\.gd|cutt\.ly|"
        r"shorte\.st|adf\.ly|lnkd\.in|po\.st|q\.gs|j\.mp"
    )

    def __init__(self, timeout=5, source="PhishTank + Legitimate URLs", version="1.0"):
        self.timeout = timeout
        self.source = source
        self.version = version
        self.total_features = len(self.FEATURE_NAMES)

    # ---------- STRING REPRESENTATIONS ----------
    def __str__(self):
        return (
            f"Phishing Feature Extractor v{self.version}\n"
            f"---------------------------------\n"
            f"Data source      : {self.source}\n"
            f"Total features   : {self.total_features}\n"
            f"Feature groups   : URL, Domain, Content\n"
        )

    # ---------- URL BASED FEATURES ----------
    def having_ip(self, url):
        try:
            host = urlparse(url).hostname
            ipaddress.ip_address(host)
            return 1
        except:
            return 0

    def have_at_sign(self, url):
        return 1 if "@" in url else 0

    def url_length(self, url):
        return 1 if len(url) >= 54 else 0

    def count_subdomain(self, url):
        try:
            hostname = urlparse(url).hostname
            if not hostname:
                return 1
            return 1 if hostname.count('.') > 2 else 0
        except:
            return 1

    def redirection(self, url):
        return 1 if url.rfind('//') > 7 else 0

    def http_domain(self, url):
        return 0 if urlparse(url).scheme == 'https' else 1

    def tiny_url(self, url):
        return 1 if self.SHORTENING_SERVICES.search(url) else 0

    def prefix_suffix(self, url):
        return 1 if '-' in urlparse(url).netloc else 0

    # ---------- DOMAIN FEATURES (combined) ----------
    def domain_info(self, domain):
        """Query WHOIS once for a domain and return info or None."""
        try:
            return whois.whois(domain)
        except:
            return None

    def domain_age(self, domain_info):
        try:
            creation = domain_info.creation_date
            expiration = domain_info.expiration_date

            if isinstance(creation, list):
                creation = creation[0]
            if isinstance(expiration, list):
                expiration = expiration[0]

            if creation is None or expiration is None:
                return 1

            age_months = (expiration - creation).days / 30
            return 1 if age_months < 6 else 0
        except:
            return 1

    def domain_end(self, domain_info):
        try:
            expiration_date = domain_info.expiration_date

            if isinstance(expiration_date, list):
                expiration_date = expiration_date[0]

            if expiration_date is None:
                return 1

            if isinstance(expiration_date, str):
                for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
                    try:
                        expiration_date = datetime.strptime(expiration_date, fmt)
                        break
                    except:
                        continue
                else:
                    return 1

            # Convert aware datetime to naive (remove timezone info)
            if expiration_date.tzinfo is not None:
                expiration_date = expiration_date.replace(tzinfo=None)

            remaining_months = (expiration_date - datetime.now()).days / 30
            return 1 if remaining_months < 6 else 0
        except:
            return 1

    # ---------- CONTENT BASED FEATURES ----------
    def get_response(self, url):
        try:
            return requests.get(url, timeout=self.timeout)
        except:
            return None

    def iframe(self, response):
        if not response:
            return 1
        return 0 if re.search(r"<iframe|frameborder", response.text, re.I) else 1

    def mouse_over(self, response):
        if not response:
            return 1
        return 1 if re.search(r"onmouseover", response.text, re.I) else 0

    def right_click(self, response):
        if not response:
            return 1
        return 0 if re.search(r"event\.button\s*==\s*2", response.text) else 1

    def forwarding(self, response):
        if not response:
            return 1
        return 1 if len(response.history) > 2 else 0

    # ---------- MAIN PIPELINE ----------
    def extract(self, url):
        features = []

        # URL-based (8)
        features.extend([
            self.having_ip(url),
            self.have_at_sign(url),
            self.url_length(url),
            self.count_subdomain(url),
            self.redirection(url),
            self.http_domain(url),
            self.tiny_url(url),
            self.prefix_suffix(url)
        ])

        # DOMAIN FEATURES
        domain = urlparse(url).netloc
        domain_info = self.domain_info(domain)
        dns_fail = 1 if domain_info is None else 0
        features.append(dns_fail)
        # Age and End only if WHOIS succeeded
        features.append(1 if dns_fail else self.domain_age(domain_info))
        features.append(1 if dns_fail else self.domain_end(domain_info))

        # CONTENT FEATURES (4)
        response = self.get_response(url)
        features.extend([
            self.iframe(response),
            self.mouse_over(response),
            self.right_click(response),
            self.forwarding(response)
        ])

        return features

    def batch_extract(self, urls):
        return [self.extract(url) for url in urls]

    # ---------- EXPLAIN FEATURES ----------
    def explain(self, url):
        """Return dictionary of feature names and values."""
        values = self.extract(url)
        return dict(zip(self.FEATURE_NAMES, values))
