"""
Define schemas
"""
from typing import Optional
from pydantic import BaseModel

class PhishingRequest(BaseModel):
    """PhishingRequest"""
    url: Optional[str] = None
    email: Optional[str] = None

class PhishingResponse(BaseModel):
    """PhishingResponse"""
    probability: float
    label: str
