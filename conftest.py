"""Root conftest.py.

Its only purpose is to anchor pytest's import-mode path insertion at the
project root, so that the analysis and test modules' absolute imports
(``from src.backend... import ...``) resolve correctly regardless of how
pytest is invoked (``pytest``, ``python -m pytest``, or from an IDE).
"""
