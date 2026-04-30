"""Core foundation package.

Contains abstract base classes (interfaces) and the domain exception hierarchy.
Nothing in this package imports from any other ``src.*`` sub-package — it is the
dependency root of the entire codebase.
"""

from src.core.exceptions import (
    ClusteringError,
    DatabaseError,
    EmbeddingError,
    IssueGroupingError,
    LLMError,
    NormalizationError,
    PipelineError,
    VectorStoreError,
)
from src.core.interfaces import (
    IClusterSearcher,
    IDataLoader,
    ILLMClient,
    IMetadataStore,
    INormalizer,
    INotifier,
    IVectorStore,
)

__all__ = [
    # Exceptions
    "IssueGroupingError",
    "EmbeddingError",
    "ClusteringError",
    "LLMError",
    "DatabaseError",
    "VectorStoreError",
    "NormalizationError",
    "PipelineError",
    # Interfaces
    "IDataLoader",
    "INormalizer",
    "IVectorStore",
    "IMetadataStore",
    "IClusterSearcher",
    "ILLMClient",
    "INotifier",
]
