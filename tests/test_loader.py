"""loader"""
from pathlib import Path
import pytest
from src.config import loader

def test_load_config_file_not_found():
    """config file"""
    # Use a path that definitely does not exist
    fake_path = Path("nonexistent_config.yaml")
    with pytest.raises(FileNotFoundError) as excinfo:
        loader.load_config(fake_path)
    assert "Config file not found" in str(excinfo.value)
