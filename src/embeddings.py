import abc
import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer

from qgenie.integrations.langchain import QGenieEmbeddings
from src.constants import QGENEIE_API_KEY
from src.execution_timer_log import execution_timer
from src.logger import AppLogger

logger = AppLogger.get_logger(__name__)


class QGenieBGEM3Embedding(Embeddings):
    name = "qgenie_embedd"

    def __init__(self):
        self.model = QGenieEmbeddings(model=self.name, api_key=QGENEIE_API_KEY)
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

    def embed(self, data: list):
        return self._retry_sync(self.model.embed_documents, data)

    def embed_documents(self, data: list[str]) -> list[list[float]]:
        return self._retry_sync(self.model.embed_documents, data)

    def embed_query(self, text: str) -> list[float]:
        return self._retry_sync(self.model.embed_query, text)

    @execution_timer
    async def aembed(self, data: list):
        return await self._retry_async(self.model.aembed_documents, data)

    @execution_timer
    async def aembed_query(self, text: str):
        return await self._retry_async(self.model.aembed_query, text)

    @execution_timer
    async def aembed_query_batch(self, text: str):
        if isinstance(text, list):
            results = [self._retry_async(self.model.aembed_query, t) for t in text]
            return await asyncio.gather(*results)
        else:
            return await self._retry_async(self.model.aembed_query, text)


class BGEM3Embeddings(object):
    _instance = None
    _model = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BGEM3Embeddings, cls).__new__(cls)
            
        from src.helpers import load_cached_model
        cls._model = load_cached_model()
        if cls._model is None:
            logger.info("Unable to find model locally downloading model")
            cls._model = SentenceTransformer("BAAI/bge-m3", cache_folder="./models", device="cpu")
        return cls._instance

    @property
    def model(self):
        return self._model

    @execution_timer
    def embed_query(self, text: str):
        return self.model.encode_query(text)

    @execution_timer
    def embed(self, data: list[str]):
        return self.model.encode_document(data)


class FallbackEmbeddings(Embeddings):
    """Embedding class that tries QGenie first and falls back to local BGEM3 if timeout occurs."""

    def __init__(self):
        self.qgenie_embeddings = QGenieBGEM3Embedding()
        self.local_embeddings = BGEM3Embeddings()
        self.timeout = 30  # 30 seconds timeout
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

    @execution_timer
    def embed(self, data: list):
        if not data:
            return []
        logger.info("Attempting to generate embeddings with QGenie")
        qgenie_result = self._run_with_timeout(self.qgenie_embeddings.embed, data)

        if qgenie_result is not None:
            logger.info("Successfully generated embeddings with QGenie")
            return qgenie_result

        logger.info("Falling back to local BGEM3 model for embeddings")
        return self.local_embeddings.embed(data)

    def embed_documents(self, data: list[str]) -> list[list[float]]:
        return self.embed(data)

    def embed_query(self, text: str) -> list[float]:
        if isinstance(text, str):
            return self.embed([text])[0]
        return self.embed([text])[0]
