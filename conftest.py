"""Root conftest.py — adds the project root to sys.path for all tests.

This ensures `import config`, `from api.main import app`, etc. all resolve
correctly regardless of where pytest is invoked from (local, CI, Docker).
"""
import sys
from pathlib import Path

# Project root = directory containing this file
sys.path.insert(0, str(Path(__file__).parent))
