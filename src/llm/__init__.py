"""LLM integration package.

Provides structured access to the QGenie / Vertex AI language model API for
cluster naming, classification, near-duplicate detection, and summarisation.

Public surface
--------------
- :class:`~src.llm.client.CustomQGenieChat` — QGenie chat client with retry
- :class:`~src.llm.client.QgenieModels` — pre-configured model instances
- :class:`~src.llm.cluster_namer.ClusterNamer` — cluster naming
- :class:`~src.llm.cluster_classifier.ClusterClassifier` — cluster classification
- :class:`~src.llm.deduplicator.Deduplicator` — near-duplicate detection and merging
"""

from src.llm.client import CustomQGenieChat, QgenieModels
from src.llm.cluster_classifier import ClusterClassifier
from src.llm.cluster_namer import ClusterNamer
from src.llm.deduplicator import Deduplicator

__all__ = [
    "CustomQGenieChat",
    "QgenieModels",
    "ClusterNamer",
    "ClusterClassifier",
    "Deduplicator",
]
