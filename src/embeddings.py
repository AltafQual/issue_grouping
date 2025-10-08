import abc
import asyncio
import time

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
    def __init__(self):
        from src.helpers import load_cached_model

        self.model = load_cached_model()

        if self.model is None:
            print("Unable to find model locally donwloading model")
            self.model = SentenceTransformer("BAAI/bge-m3", cache_folder="./models")

    @execution_timer
    def embed_query(self, text: str):
        return self.model.encode_query(text)

    @execution_timer
    def embed(self, data: list[str]):
        return self.model.encode_document(data)
