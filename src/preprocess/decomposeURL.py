from urllib.parse import urlparse
import ipaddress
import re
import whois
from datetime import datetime
import requests

def havingIP(url):
  try:
    ipaddress.ip_address(url)
    ip = 1
  except:
    ip = 0
  return ip
def haveAtSign(url):
  if "@" in url:
    at = 1    
  else:
    at = 0    
  return at
def getLength(url):
  if len(url) < 54:
    length = 0            
  else:
    length = 1            
  return length
def count_subdomain(url):
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if hostname is None:
            return 1
        parts = hostname.split('.')
        if len(parts) <= 2: # domain + TLD
            subdomain_count = 0
        else:
            subdomain_count = len(parts) - 2
        if subdomain_count <= 1:
            return 0
        else:
            return 1
    except:
        return 1
def redirection(url):
  pos = url.rfind('//')
  if pos > 6:
    if pos > 7:
      return 1
    else:
      return 0
  else:
    return 0
def httpDomain(url):
  domain = urlparse(url).netloc
  if 'https' in domain:
    return 1
  else:
    return 0
shortening_services = (
    r"bit\.ly|bitly\.com|tinyurl\.com|tinyurl\.com|"
    r"is\.gd|v\.gd|t\.co|ow\.ly|buff\.ly|"
    r"rebrand\.ly|rb\.gy|short\.io|"
    r"soo\.gd|cutt\.ly|shorte\.st|adf\.ly|"
    r"lnkd\.in|po\.st|q\.gs|j\.mp"
)
def tinyURL(url):
    match=re.search(shortening_services,url)
    if match:
        return 1
    else:
        return 0
def prefixSuffix(url):
    if '-' in urlparse(url).netloc:
        return 1
    else:
        return 0
def domainAge(domain_name):
  creation_date = domain_name.creation_date
  expiration_date = domain_name.expiration_date
  if (isinstance(creation_date,str) or isinstance(expiration_date,str)):
    try:
      creation_date = datetime.strptime(creation_date,'%Y-%m-%d')
      expiration_date = datetime.strptime(expiration_date,"%Y-%m-%d")
    except:
      return 1
  if ((expiration_date is None) or (creation_date is None)):
      return 1
  elif ((type(expiration_date) is list) or (type(creation_date) is list)):
      return 1
  else:
    ageofdomain = abs((expiration_date - creation_date).days)
    if ((ageofdomain/30) < 6):
      age = 1
    else:
      age = 0
  return age
def domainEnd(domain_name):
  expiration_date = domain_name.expiration_date
  if isinstance(expiration_date,str):
    try:
      expiration_date = datetime.strptime(expiration_date,"%Y-%m-%d")
    except:
      return 1
  if (expiration_date is None):
      return 1
  elif (type(expiration_date) is list):
      return 1
  else:
    today = datetime.now()
    end = abs((expiration_date - today).days)
    if ((end/30) < 6):
      end = 0
    else:
      end = 1
  return end
def iframe(response):
  if response == "":
      return 1
  else:
      if re.findall(r"[<iframe>|<frameBorder>]", response.text):
          return 0
      else:
          return 1
def mouseOver(response): 
  if response == "" :
    return 1
  else:
    if re.findall("<script>.+onmouseover.+</script>", response.text):
      return 1
    else:
      return 0
def rightClick(response):
  if response == "":
    return 1
  else:
    if re.findall(r"event.button ?== ?2", response.text):
      return 0
    else:
      return 1
# Checks the number of forwardings (Web_Forwards)    
def forwarding(response):
  if response == "":
    return 1
  else:
    if len(response.history) <= 2:
      return 0
    else:
      return 1

#Function to extract features
def Preprocess(url):

  features = []
  #URL-based (lexical) features (8)
  features.append(havingIP(url))
  features.append(haveAtSign(url))
  features.append(getLength(url))
  features.append(count_subdomain(url))
  features.append(redirection(url))
  features.append(httpDomain(url))
  features.append(tinyURL(url))
  features.append(prefixSuffix(url))
  
  #Domain based features (3)
  dns = 0
  try:
    domain_name = whois.whois(urlparse(url).netloc)
  except:
    dns = 1

  features.append(dns)
  features.append(1 if dns == 1 else domainAge(domain_name))
#   features.append(1 if dns == 1 else domainEnd(domain_name))
  
  # Content / page-based features Inspect HTML & Javascript (4) 
  try:
    response = requests.get(url)
  except:
    response = ""
  features.append(iframe(response))
  features.append(mouseOver(response))
  features.append(rightClick(response))
  features.append(forwarding(response))
  
  return features

def featureExtraction(label):
    feature_list = []
    for result in label:
        features = Preprocess(result)
        feature_list.append(features)
    return feature_list