"""conftest"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

# Add project root to sys.path so "import src..." works
sys.path.insert(0, str(project_root))
