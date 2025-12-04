"""
Define type of training mode
"""

from enum import Enum

class TrainingMode(str, Enum):
    """Training mode"""
    FILE = "file"
    FOLDER = "folder"

class ModelProvider(str, Enum):
    """ModelProvider"""
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
