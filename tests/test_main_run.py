"""
MAIN RUN
"""
import importlib.util
from unittest.mock import patch
import src.api.main as main

def test_main_run():
    """Run"""
    with patch("uvicorn.run") as mock_run:
        main.__name__ = "__main__"
        spec = importlib.util.spec_from_file_location("__main__", main.__file__)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        mock_run.assert_called_once()
