"""Preprocessing package — single source of truth for error-log normalisation.

All text normalisation logic lives here.  Nothing outside this package should
contain inline regex substitutions for cleaning error logs.

Public surface
--------------
- :class:`~src.preprocessing.normalizer.ErrorNormalizer` — unified normaliser
  with two modes (embedding-friendly and SPLADE-friendly)
- :class:`~src.preprocessing.log_extractor.LogExtractor` — concurrent log
  reader and error-line extractor
"""

from src.preprocessing.log_extractor import LogExtractor
from src.preprocessing.normalizer import ErrorNormalizer

__all__ = ["ErrorNormalizer", "LogExtractor"]
