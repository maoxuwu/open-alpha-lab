"""Test/dev bootstrap: make src/ importable without installation.
Production usage installs the package (`uv sync` / `pip install -e .`).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
