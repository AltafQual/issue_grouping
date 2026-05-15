"""SPLADE sparse encoder — singleton, loaded once per process.

:class:`SPLADEEncoder` wraps either the quantized
``rasyosef/splade-tiny`` model (via ``sentence-transformers``
:class:`SparseEncoder`, ~17 MB) or the full
``naver/splade-cocondenser-ensembledistil`` model (via ``transformers``,
~440 MB).  Which variant is loaded is controlled by
:attr:`~src.constants.SPLADEConfigurations.use_quantized`.

The encoder is a singleton — only one instance is created per process.
Call :meth:`SPLADEEncoder.release` on graceful shutdown to free GPU/RAM.

Layering
--------
Imports from ``src.constants``, ``src.logger``, and standard library /
PyTorch / scipy.  No circular imports.
"""

from __future__ import annotations

import gc
import os
import sys
from typing import Optional

import httpx
import numpy as np
import scipy.sparse
import torch
from sentence_transformers import SparseEncoder
from transformers import AutoModelForMaskedLM, AutoTokenizer

from src.constants import SPLADEConfigurations
from src.logger import AppLogger

logger = AppLogger().get_logger(__name__)

__all__ = ["SPLADEEncoder"]


class SPLADEEncoder:
    """Singleton SPLADE sparse encoder.

    Loaded once and kept in memory for the entire process lifetime.  Falls
    back to pure cosine similarity when the model is unavailable or disabled.

    Usage::

        enc = SPLADEEncoder()
        if enc.is_available:
            vecs = enc.encode(["error text 1", "error text 2"])
            # vecs: scipy.sparse.csr_matrix of shape (2, vocab_size)

    See :attr:`~src.constants.SPLADEConfigurations` for configuration options.
    """

    _instance: Optional["SPLADEEncoder"] = None

    def __new__(
        cls,
        model_name: str = SPLADEConfigurations.model_name,
        cache_dir: str = "./models",
    ) -> "SPLADEEncoder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_name: str = SPLADEConfigurations.model_name,
        cache_dir: str = "./models",
    ) -> None:
        if self._initialized:
            return
        self._model = None
        self._tokenizer = None
        self._available = False
        self._is_sparse_encoder = False  # True when using sentence-transformers SparseEncoder
        self._model_name = model_name
        self._cache_dir = cache_dir

        remote_url = os.getenv("SPLADE_API_URL", "").strip().rstrip("/")
        if remote_url:
            self._remote_url: Optional[str] = remote_url
            self._device = "remote"
            self._available = True
            logger.info(f"[SPLADE] Remote API mode — requests forwarded to {remote_url}")
        else:
            self._remote_url = None
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._load_model()

        self._initialized = True

    @property
    def is_available(self) -> bool:
        """``True`` when the model is loaded and ready to encode."""
        return self._available

    @property
    def device(self) -> str:
        """The device the model is running on (``'cpu'`` or ``'cuda'``)."""
        return getattr(self, "_device", "cpu")

    def encode(self, texts: list[str]) -> Optional[scipy.sparse.csr_matrix]:
        """Encode a list of texts to sparse SPLADE vectors.

        Args:
            texts: List of input strings.

        Returns:
            Scipy CSR matrix of shape ``(len(texts), vocab_size)``, or
            ``None`` if the encoder is unavailable.
        """
        if not self._available or not texts:
            return None
        if self._remote_url:
            return self._encode_remote(texts)
        try:
            if self._is_sparse_encoder:
                vecs = self._model.encode(texts, show_progress_bar=False)
                if isinstance(vecs, list):
                    dense = np.vstack([v.to_dense().cpu().numpy() for v in vecs])
                elif hasattr(vecs, "to_dense"):
                    dense = vecs.to_dense().cpu().numpy()
                else:
                    dense = np.array(vecs.cpu() if hasattr(vecs, "cpu") else vecs)
                return scipy.sparse.csr_matrix(dense)
            else:
                with torch.no_grad():
                    inputs = self._tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                    logits = self._model(**inputs).logits  # (batch, seq_len, vocab_size)
                    sparse_vecs = torch.log(1 + torch.relu(logits)).max(dim=1).values.cpu()  # (batch, vocab_size)
                return scipy.sparse.csr_matrix(sparse_vecs.numpy())
        except Exception as exc:
            logger.error(f"[SPLADE] Encoding failed: {exc}")
            return None

    def _encode_remote(self, texts: list[str]) -> Optional[scipy.sparse.csr_matrix]:
        """Call the remote SPLADE API and reconstruct a CSR matrix from the response."""
        try:
            resp = httpx.post(
                f"{self._remote_url}/api/splade/encode/",
                json={"texts": texts},
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != 200:
                logger.warning(f"[SPLADE] Remote API error: {data.get('error')}")
                return None
            embeddings = np.array(data["embeddings"], dtype=np.float32)
            return scipy.sparse.csr_matrix(embeddings)
        except Exception as exc:
            logger.warning(f"[SPLADE] Remote encode failed ({self._remote_url}): {exc}")
            return None

    def encode_single(self, text: str) -> Optional[scipy.sparse.csr_matrix]:
        """Encode a single text string.

        Args:
            text: Input string.

        Returns:
            Scipy CSR matrix of shape ``(1, vocab_size)``, or ``None``.
        """
        return self.encode([text])

    @classmethod
    def release(cls) -> None:
        """Unload the model from RAM and reset the singleton.

        Also clears the in-memory cluster vector cache in
        :class:`~src.clustering.hybrid_matcher.HybridSPLADEMatcher`.
        Call this on graceful application shutdown (e.g. FastAPI lifespan).
        """
        inst = cls._instance
        if inst is None:
            return

        had_local_model = inst._model is not None
        if had_local_model:
            del inst._model
            inst._model = None
        if inst._tokenizer is not None:
            del inst._tokenizer
            inst._tokenizer = None
        inst._available = False
        inst._initialized = False
        cls._instance = None

        # Clear the in-memory cluster vector cache in HybridSPLADEMatcher
        hm_mod = sys.modules.get("src.clustering.hybrid_matcher")
        if hm_mod is not None:
            hm_mod.HybridSPLADEMatcher._cluster_vec_cache.clear()

        if had_local_model:
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def _load_model(self) -> None:
        if not SPLADEConfigurations.enabled:
            logger.info("[SPLADE] Disabled via SPLADEConfigurations.enabled=False; using pure cosine.")
            self._available = False
            return
        if SPLADEConfigurations.use_quantized:
            self._load_quantized_model()
        else:
            self._load_full_model()

    def _load_quantized_model(self) -> None:
        """Load ``rasyosef/splade-tiny`` via sentence-transformers (~17 MB)."""
        model_id = SPLADEConfigurations.quantized_model_name
        try:
            logger.info(f"[SPLADE] Loading quantized model: {model_id}")
            self._model = SparseEncoder(model_id, cache_folder=self._cache_dir, device=self._device)
            self._is_sparse_encoder = True
            self._available = True
            self._model_name = model_id
            logger.info(f"[SPLADE] Quantized model loaded: {model_id}")
        except Exception as exc:
            logger.warning(f"[SPLADE] Quantized model unavailable ({exc}); falling back to pure cosine.")
            self._available = False

    def _load_full_model(self) -> None:
        """Load ``naver/splade-cocondenser-ensembledistil`` via transformers (~440 MB)."""
        try:
            model_folder = f"models--{self._model_name.replace('/', '--')}"
            snapshots_path = os.path.join(self._cache_dir, model_folder, "snapshots")
            if os.path.exists(snapshots_path):
                snapshots = os.listdir(snapshots_path)
                if snapshots:
                    local_path = os.path.join(snapshots_path, snapshots[0])
                    logger.info(f"[SPLADE] Loading from local cache: {local_path}")
                    self._tokenizer = AutoTokenizer.from_pretrained(local_path)
                    self._model = AutoModelForMaskedLM.from_pretrained(
                        local_path, low_cpu_mem_usage=True, dtype=torch.float32
                    )
                    self._model.to(self._device)
                    self._model.eval()
                    self._available = True
                    return

            logger.info(f"[SPLADE] Loading from HuggingFace: {self._model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, cache_dir=self._cache_dir)
            self._model = AutoModelForMaskedLM.from_pretrained(
                self._model_name, cache_dir=self._cache_dir, low_cpu_mem_usage=True, dtype=torch.float32
            )
            self._model.to(self._device)
            self._model.eval()
            self._available = True
            logger.info(f"[SPLADE] Full model loaded: {self._model_name}")
        except Exception as exc:
            logger.warning(f"[SPLADE] Model unavailable ({exc}); SPLADE disabled, using pure cosine.")
            self._available = False
