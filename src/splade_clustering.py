"""
SPLADE-enhanced clustering utilities for error log grouping.
- ErrorNormalizer: strips noise from error logs before SPLADE encoding
- SPLADEEncoder: encodes text to sparse vectors via SPLADE model (singleton, stays loaded)
- HybridSPLADEMatcher: combines cosine + SPLADE scores for matching (in-memory cluster cache)
- ClusterRanker: ranks cluster members by embedding cosine similarity
- ClusterCohesionAnalyzer: detects loose/poorly-formed clusters
"""

import gc
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse
import torch
from sentence_transformers import SparseEncoder
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.constants import DataFrameKeys, FaissConfigurations, SPLADEConfigurations
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)


class ErrorNormalizer:
    """Strips noise from error logs to improve SPLADE matching accuracy."""

    _PATTERNS = [
        # CamelCase → "Camel Case" (before other patterns so SPLADE tokenizer sees words)
        (re.compile(r"([a-z])([A-Z])"), r"\1 \2"),
        # snake_case → "snake case"
        (re.compile(r"_"), " "),
        # "Limiting Reason To 3000 chars|" prefix
        (re.compile(r"Limiting Reason To \d+ chars\|", re.IGNORECASE), ""),
        # Leading exit-code prefix like "9: ", "139: "
        (re.compile(r"^\d+:\s+"), ""),
        # Absolute paths
        (re.compile(r"/prj/[^\s,;]+"), "<PATH>"),
        (re.compile(r"/tmp/AiswTest_[A-Za-z0-9_]+[^\s,;]*"), "<TMPPATH>"),
        (re.compile(r"/data/local/[^\s,;]+"), "<PATH>"),
        (re.compile(r"/[a-zA-Z][a-zA-Z0-9_\-/]+\.[a-zA-Z]{1,5}(?=\s|,|;|$)"), "<FILEPATH>"),
        # PIDs
        (re.compile(r"pid:\s*\d+", re.IGNORECASE), ""),
        # Version strings like v2.36.0.250610144245_123137-auto
        (re.compile(r"v\d+\.\d+\.\d+\.\d+[_\-][a-zA-Z0-9_\-]+"), "<VERSION>"),
        # Result/Run numbers
        (re.compile(r"\bResult_\d+\b"), "<RESULT>"),
        (re.compile(r"\bRun\d+\b"), "<RUN>"),
        # Random temp dir names AiswTest_Abc123
        (re.compile(r"AiswTest_[A-Za-z0-9]+"), "<TMPDIR>"),
        # Timing values like "1234.56ms" or "within 3.14 secs"
        (re.compile(r"\b\d+\.\d+ms\b"), "<TIMING>"),
        (re.compile(r"within\s+[\d\.]+\s+secs?", re.IGNORECASE), "within <TIME> secs"),
        # Pure hex addresses
        (re.compile(r"\b0x[0-9a-fA-F]{4,}\b"), "<ADDR>"),
        # Long sequences of digits (timestamps, port numbers, etc.)
        (re.compile(r"\b\d{6,}\b"), "<NUM>"),
        # Repeated whitespace
        (re.compile(r"\s+"), " "),
    ]

    def normalize(self, text: str) -> str:
        """Return noise-stripped version of error text."""
        text = text.strip()
        for pattern, replacement in self._PATTERNS:
            text = pattern.sub(replacement, text)
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """Normalize then tokenize into lowercase words (min length 2)."""
        normalized = self.normalize(text)
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9]{1,}", normalized)
        return [t.lower() for t in tokens if len(t) >= 2]


class SPLADEEncoder:
    """
    Singleton SPLADE encoder. Loaded once and kept in memory for the process lifetime.

    When SPLADEConfigurations.use_quantized=True (default):
        Uses rasyosef/splade-tiny via sentence-transformers SparseEncoder (~17 MB).

    When use_quantized=False:
        Uses naver/splade-cocondenser-ensembledistil via transformers (~440 MB).

    Returns scipy.sparse.csr_matrix (batch_size, vocab_size).
    Falls back gracefully if model is unavailable (is_available returns False).
    """

    _instance = None

    def __new__(cls, model_name: str = SPLADEConfigurations.model_name, cache_dir: str = "./models"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = SPLADEConfigurations.model_name, cache_dir: str = "./models"):
        if self._initialized:
            return
        self._model = None
        self._tokenizer = None
        self._available = False
        self._is_sparse_encoder = False  # True when using sentence-transformers SparseEncoder
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._load_model()
        self._initialized = True

    def _load_model(self) -> None:
        if not SPLADEConfigurations.enabled:
            logger.info("[SPLADE] Disabled via SPLADEConfigurations.enabled=False. Using pure cosine.")
            self._available = False
            return

        if SPLADEConfigurations.use_quantized:
            self._load_quantized_model()
        else:
            self._load_full_model()

    def _load_quantized_model(self) -> None:
        """Load rasyosef/splade-tiny via sentence-transformers SparseEncoder (~17 MB)."""
        model_id = SPLADEConfigurations.quantized_model_name
        try:
            logger.info(f"[SPLADE] Loading quantized model: {model_id}")
            self._model = SparseEncoder(model_id, cache_folder=self._cache_dir)
            self._is_sparse_encoder = True
            self._available = True
            logger.info(f"[SPLADE] Quantized model loaded: {model_id}")
        except Exception as e:
            logger.warning(f"[SPLADE] Quantized model unavailable ({e}). Falling back to pure cosine.")
            self._available = False

    def _load_full_model(self) -> None:
        """Load naver/splade-cocondenser-ensembledistil via transformers (~440 MB)."""
        try:
            model_folder = f"models--{self._model_name.replace('/', '--')}"
            snapshots_path = os.path.join(self._cache_dir, model_folder, "snapshots")
            if os.path.exists(snapshots_path):
                snapshots = os.listdir(snapshots_path)
                if snapshots:
                    local_path = os.path.join(snapshots_path, snapshots[0])
                    logger.info(f"[SPLADE] Loading model from local cache: {local_path}")
                    self._tokenizer = AutoTokenizer.from_pretrained(local_path)
                    self._model = AutoModelForMaskedLM.from_pretrained(
                        local_path, low_cpu_mem_usage=True, dtype=torch.float32
                    )
                    self._model.eval()
                    self._available = True
                    logger.info(f"[SPLADE] Model loaded successfully from {local_path}")
                    return

            logger.info(f"[SPLADE] Loading model from HuggingFace: {self._model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, cache_dir=self._cache_dir)
            self._model = AutoModelForMaskedLM.from_pretrained(
                self._model_name, cache_dir=self._cache_dir, low_cpu_mem_usage=True, dtype=torch.float32
            )
            self._model.eval()
            self._available = True
            logger.info(f"[SPLADE] Model loaded successfully: {self._model_name}")
        except Exception as e:
            logger.warning(f"[SPLADE] Model unavailable ({e}). SPLADE scoring disabled; falling back to pure cosine.")
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def encode(self, texts: List[str]) -> Optional[scipy.sparse.csr_matrix]:
        """
        Encode a list of texts to sparse SPLADE vectors.

        Returns scipy.sparse.csr_matrix of shape (len(texts), vocab_size),
        or None if model is unavailable.
        """
        if not self._available or not texts:
            return None
        try:
            if self._is_sparse_encoder:
                # SparseEncoder returns torch.sparse_coo_tensor objects.
                # Convert to dense numpy then to scipy CSR.
                vecs = self._model.encode(texts, show_progress_bar=False)
                if isinstance(vecs, list):
                    dense = np.vstack([v.to_dense().numpy() for v in vecs])
                elif hasattr(vecs, "to_dense"):
                    dense = vecs.to_dense().numpy()
                else:
                    dense = np.array(vecs)
                return scipy.sparse.csr_matrix(dense)
            else:
                with torch.no_grad():
                    inputs = self._tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    logits = self._model(**inputs).logits  # (batch, seq_len, vocab_size)
                    sparse_vecs = torch.log(1 + torch.relu(logits)).max(dim=1).values  # (batch, vocab_size)
                    sparse_np = sparse_vecs.numpy()
                return scipy.sparse.csr_matrix(sparse_np)
        except Exception as e:
            logger.error(f"[SPLADE] Encoding failed: {e}")
            return None

    def encode_single(self, text: str) -> Optional[scipy.sparse.csr_matrix]:
        """Encode a single text string."""
        return self.encode([text])

    @classmethod
    def release(cls) -> None:
        """
        Unload the SPLADE model from RAM and reset the singleton.
        Also clears the HybridSPLADEMatcher in-memory cluster vector cache.

        Call this on graceful application shutdown.
        """
        inst = cls._instance
        if inst is None:
            logger.info("[SPLADE] SPLADEEncoder.release() called but no instance loaded — nothing to do")
            return

        model_name = getattr(inst, "_model_name", "unknown")
        if inst._model is not None:
            del inst._model
            inst._model = None
        if inst._tokenizer is not None:
            del inst._tokenizer
            inst._tokenizer = None
        inst._available = False
        inst._initialized = False
        cls._instance = None

        # Clear the in-memory cluster vector cache in HybridSPLADEMatcher
        HybridSPLADEMatcher._cluster_vec_cache.clear()

        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        logger.info(f"[SPLADE] SPLADEEncoder released from memory (model={model_name})")


class HybridSPLADEMatcher:
    """
    Combines cosine similarity (embedding) and SPLADE sparse dot-product scores.

    Cluster SPLADE vectors are computed from cluster names on first use and cached
    in memory for the process lifetime — no disk I/O or EMA updates.

    alpha: weight for embedding cosine score (default 0.55)
    beta:  weight for SPLADE score (default 0.45)

    Falls back to pure cosine if the SPLADE encoder is unavailable.
    """

    # Class-level cache shared across all instances: {type_: {cluster_name: sparse_vec}}
    _cluster_vec_cache: Dict[str, Dict[str, scipy.sparse.csr_matrix]] = {}

    def __init__(
        self,
        alpha: float = SPLADEConfigurations.hybrid_alpha,
        beta: float = SPLADEConfigurations.hybrid_beta,
    ):
        self.alpha = alpha
        self.beta = beta

    def _get_encoder(self) -> Optional["SPLADEEncoder"]:
        enc = SPLADEEncoder()
        return enc if enc.is_available else None

    @staticmethod
    def _normalize_scores(raw: np.ndarray) -> np.ndarray:
        """Min-max normalize scores to [0, 1]."""
        min_s, max_s = raw.min(), raw.max()
        if max_s - min_s < 1e-9:
            return np.zeros_like(raw)
        return (raw - min_s) / (max_s - min_s)

    def _get_cluster_matrix(
        self, type_: str, cluster_names: List[str], enc: "SPLADEEncoder"
    ) -> Optional[scipy.sparse.csr_matrix]:
        """
        Return a (len(cluster_names), vocab_size) CSR matrix of SPLADE vectors.
        Unseen cluster names are encoded on-the-fly and cached in memory.
        Returns None if any cluster could not be encoded.
        """
        cache = HybridSPLADEMatcher._cluster_vec_cache.setdefault(type_, {})

        new_names = [n for n in cluster_names if n not in cache]
        if new_names:
            vecs = enc.encode(new_names)
            if vecs is not None:
                for i, name in enumerate(new_names):
                    cache[name] = vecs[i]  # (1, vocab_size) slice

        rows = [cache.get(n) for n in cluster_names]
        if any(r is None for r in rows):
            return None
        return scipy.sparse.vstack(rows, format="csr")

    def search(
        self,
        type_: str,
        query: str,
        query_embedding: np.ndarray,
        centroids: np.ndarray,
        cluster_names: List[str],
        threshold: float,
    ) -> Tuple[int, float]:
        """
        Find best matching cluster using hybrid scoring.
        Returns (best_idx, best_score). Returns (-1, best_score) if below threshold.
        """
        cosine_scores = cosine_similarity([query_embedding], centroids)[0]
        scores = cosine_scores
        scoring_mode = "pure cosine"

        enc = self._get_encoder()
        if enc is not None:
            query_vec = enc.encode_single(query)
            cluster_matrix = self._get_cluster_matrix(type_, cluster_names, enc)
            if query_vec is not None and cluster_matrix is not None:
                raw = (cluster_matrix @ query_vec.T).toarray().flatten()
                splade_scores = self._normalize_scores(raw)
                scores = self.alpha * cosine_scores + self.beta * splade_scores
                scoring_mode = f"hybrid (α={self.alpha}, β={self.beta})"

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= threshold:
            logger.info(
                f"[HybridSPLADE] type={type_} | {scoring_mode} | "
                f"MATCH → '{cluster_names[best_idx]}' (score={best_score:.3f}, threshold={threshold})"
            )
            return best_idx, best_score

        logger.debug(
            f"[HybridSPLADE] type={type_} | {scoring_mode} | "
            f"NO MATCH (best_score={best_score:.3f} < threshold={threshold})"
        )
        return -1, best_score

    def batch_search(
        self,
        type_: str,
        queries: List[str],
        query_embeddings: np.ndarray,
        centroids: np.ndarray,
        cluster_names: List[str],
        threshold: float,
    ) -> Tuple[List[int], List[float]]:
        """
        Batch hybrid search.
        Returns (best_indices, best_scores). Index -1 means no match found.
        """
        cosine_matrix = cosine_similarity(query_embeddings, centroids)  # (N, C)
        score_matrix = cosine_matrix
        scoring_mode = "pure cosine (SPLADE unavailable)"

        enc = self._get_encoder()
        if enc is not None:
            query_vecs = enc.encode(queries)
            cluster_matrix = self._get_cluster_matrix(type_, cluster_names, enc)
            if query_vecs is not None and cluster_matrix is not None:
                # (N_queries, vocab) @ (vocab, N_clusters) → (N_queries, N_clusters)
                raw_splade = (query_vecs @ cluster_matrix.T).toarray()
                splade_matrix = np.vstack([self._normalize_scores(row) for row in raw_splade])
                score_matrix = self.alpha * cosine_matrix + self.beta * splade_matrix
                scoring_mode = f"hybrid (α={self.alpha}, β={self.beta})"

        logger.info(
            f"[HybridSPLADE] Batch search for type={type_} | {scoring_mode} | "
            f"{len(queries)} queries against {len(cluster_names)} clusters"
        )

        best_indices = []
        best_scores = []
        for row in score_matrix:
            best_idx = int(np.argmax(row))
            best_score = float(row[best_idx])
            best_indices.append(best_idx if best_score >= threshold else -1)
            best_scores.append(best_score)

        matched = sum(1 for i in best_indices if i >= 0)
        logger.info(
            f"[HybridSPLADE] Batch results for type={type_}: "
            f"{matched}/{len(queries)} matched (threshold={threshold})"
        )
        return best_indices, best_scores


class ClusterRanker:
    """
    Ranks members within each cluster by cosine similarity to the cluster centroid.

    Uses the existing embeddings column — no extra SPLADE inference.

    Adds columns:
      - rank: 1 = most representative (nearest to centroid)
      - representativeness_score: [0, 1] (1 = closest to centroid)
      - is_core_member: True for top 50% (configurable)
    """

    def __init__(self, core_percentile: float = SPLADEConfigurations.core_member_percentile):
        self.core_percentile = core_percentile

    def rank_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rank, representativeness_score, is_core_member columns.
        Processes each cluster independently using precomputed embeddings.
        """
        df = df.copy()
        df["rank"] = -1
        df["representativeness_score"] = 0.0
        df["is_core_member"] = False

        emb_col = DataFrameKeys.embeddings_key
        cluster_col = DataFrameKeys.cluster_name

        if emb_col not in df.columns or cluster_col not in df.columns:
            logger.warning("[ClusterRanker] Missing required columns, skipping ranking")
            return df

        cluster_groups = list(df.groupby(cluster_col))
        logger.info(f"[ClusterRanker] Ranking {len(cluster_groups)} clusters ({len(df)} total rows)")

        for cluster_name, group in cluster_groups:
            valid_mask = group[emb_col].notna()
            valid_group = group[valid_mask]
            if valid_group.empty:
                continue

            embeddings = np.vstack(valid_group[emb_col].values)
            centroid = embeddings.mean(axis=0, keepdims=True)
            sims = cosine_similarity(embeddings, centroid).flatten()

            min_s, max_s = sims.min(), sims.max()
            if max_s - min_s > 1e-9:
                norm_scores = (sims - min_s) / (max_s - min_s)
            else:
                norm_scores = np.ones(len(sims))

            n = len(sims)
            ranks = n - np.argsort(np.argsort(sims))  # descending rank (1 = best)
            threshold = np.percentile(norm_scores, (1 - self.core_percentile) * 100)

            for iloc_pos, score, rank in zip(valid_group.index, norm_scores, ranks):
                df.at[iloc_pos, "rank"] = int(rank)
                df.at[iloc_pos, "representativeness_score"] = float(score)
                df.at[iloc_pos, "is_core_member"] = bool(score >= threshold)

            logger.debug(f"[ClusterRanker] Cluster '{cluster_name}': {n} members, " f"core_threshold={threshold:.3f}")

        total_core = df["is_core_member"].sum()
        logger.info(f"[ClusterRanker] Done: {total_core}/{len(df)} rows marked as core members")
        return df


class ClusterCohesionAnalyzer:
    """
    Detects loose clusters using intra-cluster embedding cosine similarity.

    Uses the existing embeddings column — no extra SPLADE inference.

    Adds columns:
      - cluster_cohesion_score: mean pairwise cosine similarity [0, 1]
      - is_loose_cluster: True if cohesion < LOW_COHESION_THRESHOLD and size > 5
    """

    LOW_COHESION_THRESHOLD = SPLADEConfigurations.low_cohesion_threshold

    def analyze_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cluster_cohesion_score and is_loose_cluster columns.
        """
        df = df.copy()
        df["cluster_cohesion_score"] = 1.0
        df["is_loose_cluster"] = False

        emb_col = DataFrameKeys.embeddings_key
        cluster_col = DataFrameKeys.cluster_name

        if emb_col not in df.columns or cluster_col not in df.columns:
            logger.warning("[CohesionAnalyzer] Missing required columns, skipping cohesion analysis")
            return df

        cluster_groups = list(df.groupby(cluster_col))
        logger.info(f"[CohesionAnalyzer] Analyzing cohesion for {len(cluster_groups)} clusters")

        loose_clusters = []
        for cluster_name, group in cluster_groups:
            valid_mask = group[emb_col].notna()
            valid_group = group[valid_mask]
            n = len(valid_group)

            if n < 2:
                # Single-member cluster — cohesion is trivially 1.0
                continue

            embeddings = np.vstack(valid_group[emb_col].values)
            sim_matrix = cosine_similarity(embeddings)  # (n, n)

            # Mean pairwise cosine (exclude diagonal self-similarity)
            off_diag = sim_matrix[~np.eye(n, dtype=bool)]
            cohesion = float(off_diag.mean()) if len(off_diag) > 0 else 1.0
            is_loose = cohesion < self.LOW_COHESION_THRESHOLD and n > 5

            df.loc[group.index, "cluster_cohesion_score"] = cohesion
            df.loc[group.index, "is_loose_cluster"] = is_loose

            logger.debug(
                f"[CohesionAnalyzer] Cluster '{cluster_name}': " f"size={n}, cohesion={cohesion:.3f}, loose={is_loose}"
            )
            if is_loose:
                loose_clusters.append(cluster_name)

        if loose_clusters:
            logger.warning(
                f"[CohesionAnalyzer] {len(loose_clusters)} loose clusters detected "
                f"(cohesion < {self.LOW_COHESION_THRESHOLD}): {loose_clusters}"
            )
        else:
            logger.info(
                f"[CohesionAnalyzer] All {len(cluster_groups)} clusters have acceptable cohesion "
                f"(threshold={self.LOW_COHESION_THRESHOLD})"
            )

        return df
