"""QGenie embedding provider with BGE-M3 fallback.

Provides three classes:

:class:`QGenieBGEM3Embedding`
    Thin wrapper around :class:`qgenie.integrations.langchain.QGenieEmbeddings`
    with retry logic (3 attempts, 5 s sleep between attempts) for both sync
    and async paths.

:class:`FallbackEmbeddings`
    Production embedding provider.  Calls QGenie first; on timeout or repeated
    500 errors automatically falls back to the local BGE-M3 model.  Handles
    large batches by splitting into sub-batches and reducing sub-batch size on
    HTTP 500 errors.

Layering
--------
Imports from ``src.embeddings.base``, ``src.embeddings.bge_provider``,
``src.utils.timer``, ``src.constants``, ``src.logger``.
"""

from __future__ import annotations

import asyncio
import threading
import time

from langchain.embeddings.base import Embeddings
from qgenie.integrations.langchain import QGenieEmbeddings

from src.constants import QGENEIE_API_KEY
from src.embeddings.base import EmbeddingProvider
from src.embeddings.bge_provider import BGEM3Embeddings
from src.logger import AppLogger
from src.utils.timer import execution_timer

logger = AppLogger().get_logger(__name__)

__all__ = ["QGenieBGEM3Embedding", "FallbackEmbeddings"]


class QGenieBGEM3Embedding(Embeddings):
    """QGenie embeddings wrapper with retry logic.

    Wraps :class:`qgenie.integrations.langchain.QGenieEmbeddings` and adds
    3-attempt retry with 5 s back-off for both synchronous and asynchronous
    embedding calls.

    Args:
        api_key: QGenie API key.  Defaults to the value of the
            ``QGENEIE_API_KEY`` constant.

    Example::

        provider = QGenieBGEM3Embedding()
        vectors = provider.embed(["error log text"])
    """

    name = "qgenie_embedd"

    def __init__(self, api_key: str = QGENEIE_API_KEY) -> None:
        self.model = QGenieEmbeddings(model=self.name, api_key=api_key)
        super().__init__()

    def _retry_sync(self, func, *args, **kwargs):
        for attempt in range(3):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise Exception(f"Failed to generate embeddings after 3 attempts: {e}")
                time.sleep(5)

    async def _retry_async(self, func, *args, **kwargs):
        for attempt in range(3):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Async attempt {attempt + 1} failed: {e}")
                if attempt == 2:
                    raise Exception(
                        f"Failed to generate embeddings after 3 attempts: {e} \n for inputs: {args} {kwargs}"
                    )
                await asyncio.sleep(5)

    def embed(self, data: list) -> list:
        """Embed documents in batches of 50.

        Args:
            data: List of text strings.

        Returns:
            List of embedding vectors.
        """
        logger.info(f"Starting embedding process for {len(data)} documents")
        batch_size = 50
        results = []

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            batch_end = min(i + batch_size, len(data))
            logger.info(f"Processing batch: documents {i} to {batch_end - 1}")
            batch_result = self._retry_sync(self.model.embed_documents, batch)
            logger.info(f"Completed batch with {len(batch)} documents")
            results.extend(batch_result)

        logger.info(f"Embedding process completed, returning {len(results)} embeddings")
        return results

    def embed_documents(self, data: list[str]) -> list[list[float]]:
        """Langchain-compatible sync batch embed."""
        return self._retry_sync(self.model.embed_documents, data)

    def embed_query(self, text: str) -> list[float]:
        """Langchain-compatible sync single-text embed."""
        return self._retry_sync(self.model.embed_query, text)

    @execution_timer
    def embed_without_retry(self, data: list) -> list:
        """Embed without retry (used internally for timeout-controlled calls)."""
        return self.model.embed_documents(data)

    @execution_timer
    async def aembed_without_retry(self, data: list) -> list:
        """Async embed without retry (used internally for timeout-controlled calls)."""
        return await self.model.aembed_documents(data)

    @execution_timer
    async def aembed(self, data: list) -> list:
        """Async embed with retry."""
        return await self._retry_async(self.model.aembed_documents, data)

    @execution_timer
    async def aembed_query(self, text: str) -> list[float]:
        """Async single-query embed with retry."""
        return await self._retry_async(self.model.aembed_query, text)

    @execution_timer
    async def aembed_query_batch(self, text) -> list:
        """Async embed a query or list of queries."""
        if isinstance(text, list):
            results = [self._retry_async(self.model.aembed_query, t) for t in text]
            return await asyncio.gather(*results)
        return await self._retry_async(self.model.aembed_query, text)


class FallbackEmbeddings(Embeddings, EmbeddingProvider):
    """Production embedding provider — QGenie first, local BGE-M3 on failure.

    Attempts to embed using :class:`QGenieBGEM3Embedding`.  On timeout (default
    600 s) or repeated HTTP 500 errors, automatically falls back to the local
    :class:`~src.embeddings.bge_provider.BGEM3Embeddings` model.

    Large batches are split into 500-document sub-batches.  Sub-batch size is
    reduced by 5 on each 500 error until it reaches 1.

    Args:
        timeout: Per-batch timeout in seconds for QGenie calls.  Defaults to
            600.

    Example::

        embedder = FallbackEmbeddings()
        vectors = await embedder.aembed(texts)
    """

    def __init__(self, timeout: int = 600) -> None:
        self.qgenie_embeddings = QGenieBGEM3Embedding()
        self.timeout = timeout
        super().__init__()

    def _run_with_timeout(self, func, *args, **kwargs):
        result = [None]
        exception = [None]
        completed = [False]

        def target():
            try:
                result[0] = func(*args, **kwargs)
                completed[0] = True
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout)

        if completed[0]:
            return result[0]
        elif exception[0]:
            logger.error(f"Error in QGenie embedding: {exception[0]}")
            return None
        else:
            logger.warning(f"QGenie embedding timed out after {self.timeout} seconds")
            return None

    def _try_embed_sub_batch(self, sub_batch: list) -> list:
        """Embed a single sub-batch using a thread with timeout.

        Args:
            sub_batch: Subset of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            TimeoutError: If QGenie does not respond within ``self.timeout`` s.
            Exception: On any other embedding failure.
        """
        result = [None]
        exception = [None]
        completed = [False]

        def target(b=sub_batch):
            try:
                result[0] = self.qgenie_embeddings.embed_without_retry(b)
                completed[0] = True
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout)

        if completed[0]:
            return result[0]
        elif exception[0] is not None:
            raise exception[0]
        else:
            raise TimeoutError(f"QGenie embedding timed out after {self.timeout} seconds")

    def _embed_batch_with_size_reduction(self, batch: list) -> list:
        """Embed a batch, reducing sub-batch size by 5 on HTTP 500 errors.

        Args:
            batch: Full batch of texts.

        Returns:
            List of embedding vectors for the entire batch.

        Raises:
            Exception: If embedding fails even with batch size 1.
        """
        current_batch_size = len(batch)
        while current_batch_size >= 1:
            sub_results = []
            got_500 = False
            for j in range(0, len(batch), current_batch_size):
                sub_batch = batch[j : j + current_batch_size]
                try:
                    sub_results.extend(self._try_embed_sub_batch(sub_batch))
                except Exception as e:
                    err_str = str(e)
                    if "500" in err_str or "internal server error" in err_str.lower():
                        if current_batch_size == 1:
                            raise Exception(f"Failed to generate embeddings even with batch size 1: {e}")
                        logger.warning(f"500 error with batch size {current_batch_size}, reducing by 5 and retrying")
                        got_500 = True
                        break
                    else:
                        raise
            if not got_500:
                return sub_results
            current_batch_size = max(1, current_batch_size - 5)

    @execution_timer
    def embed(self, data: list) -> list:
        """Synchronously embed a list of texts.

        Processes data in 500-document batches.  Falls back to local BGE-M3
        if QGenie fails.

        Args:
            data: List of text strings.

        Returns:
            List of embedding vectors.
        """
        if not data:
            return []
        results = []
        logger.info(f"Attempting to generate embeddings with QGenie: length of data: {len(data)}")
        batch_size = 500

        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            batch_end = min(i + batch_size, len(data))
            logger.info(f"Processing batch: documents {i} to {batch_end - 1}")
            batch_result = self._embed_batch_with_size_reduction(batch)
            results.extend(batch_result)
            logger.info(f"Completed batch with {len(batch)} documents")

        return results

    async def _try_aembed_sub_batch(self, sub_batch: list) -> list:
        return await asyncio.wait_for(
            self.qgenie_embeddings.aembed_without_retry(sub_batch),
            timeout=120,
        )

    async def _aembed_batch_with_size_reduction(self, batch: list, batch_start: int) -> list:
        current_batch_size = len(batch)
        while current_batch_size >= 1:
            sub_results = []
            got_500 = False
            for j in range(0, len(batch), current_batch_size):
                sub_batch = batch[j : j + current_batch_size]
                try:
                    sub_results.extend(await self._try_aembed_sub_batch(sub_batch))
                except Exception as e:
                    err_str = str(e)
                    if "500" in err_str or "internal server error" in err_str.lower():
                        if current_batch_size == 1:
                            raise Exception(f"Failed to generate embeddings even with batch size 1: {e}")
                        logger.warning(f"500 error with batch size {current_batch_size}, reducing by 5 and retrying")
                        got_500 = True
                        break
                    else:
                        raise
            if not got_500:
                return sub_results
            current_batch_size = max(1, current_batch_size - 5)

    @execution_timer
    async def aembed(self, data: list) -> list:
        """Asynchronously embed a list of texts.

        Processes data in 500-document batches concurrently.  Falls back to
        local BGE-M3 on any QGenie failure.

        Args:
            data: List of text strings.

        Returns:
            List of embedding vectors.
        """
        if not data:
            return []
        try:
            logger.info(f"Attempting to generate embeddings with QGenie: length of data: {len(data)}")
            batch_size = 500
            batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

            async def process_batch(batch: list, index: int) -> list:
                batch_start = index * batch_size
                batch_end = min(batch_start + batch_size, len(data))
                logger.info(f"Processing batch: documents {batch_start} to {batch_end - 1}")
                result = await self._aembed_batch_with_size_reduction(batch, batch_start)
                logger.info(f"Completed batch with {len(batch)} documents")
                return result

            batch_tasks = [process_batch(batch, i) for i, batch in enumerate(batches)]
            batch_results = await asyncio.gather(*batch_tasks)

            results = []
            for batch_result in batch_results:
                results.extend(batch_result)
            return results
        except Exception as e:
            logger.error(f"QGenie aembed failed ({type(e).__name__}: {e}). Falling back to local BGEM3.")
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, BGEM3Embeddings().embed, data)

    def embed_documents(self, data: list[str]) -> list[list[float]]:
        """Langchain-compatible alias for :meth:`embed`."""
        return self.embed(data)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed([text])[0]
