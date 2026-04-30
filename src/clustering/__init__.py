"""Clustering package — vector storage and similarity search.

Splits the old 665-line ``CustomEmbeddingCluster`` god-class into focused
components with clear single responsibilities.

Public surface
--------------
- :class:`~src.clustering.vector_store.VectorStore` — centroids.npy R/W
- :class:`~src.clustering.metadata_store.MetadataStore` — metadata.json R/W
- :class:`~src.clustering.searcher.ClusterSearcher` — cosine + hybrid search
- :class:`~src.clustering.splade_encoder.SPLADEEncoder` — sparse encoder singleton
- :class:`~src.clustering.hybrid_matcher.HybridSPLADEMatcher` — hybrid scorer
- :class:`~src.clustering.ranker.ClusterRanker` — representativeness ranking
- :class:`~src.clustering.ranker.ClusterCohesionAnalyzer` — cohesion detection
- :class:`~src.clustering.hdbscan_clusterer.HDBSCANClusterer` — HDBSCAN wrapper
"""

from src.clustering.hdbscan_clusterer import HDBSCANClusterer
from src.clustering.hybrid_matcher import HybridSPLADEMatcher
from src.clustering.metadata_store import MetadataStore
from src.clustering.ranker import ClusterCohesionAnalyzer, ClusterRanker
from src.clustering.searcher import ClusterSearcher
from src.clustering.splade_encoder import SPLADEEncoder
from src.clustering.vector_store import VectorStore

__all__ = [
    "VectorStore",
    "MetadataStore",
    "ClusterSearcher",
    "SPLADEEncoder",
    "HybridSPLADEMatcher",
    "ClusterRanker",
    "ClusterCohesionAnalyzer",
    "HDBSCANClusterer",
]
