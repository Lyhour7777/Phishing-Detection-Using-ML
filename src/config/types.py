"""
Define type of training mode
"""

from enum import Enum

class TrainingMode(str, Enum):
    """Training mode"""
    FILE = "file"
    FOLDER = "folder"
